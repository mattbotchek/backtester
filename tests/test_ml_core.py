import builtins
import pickle
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yfinance

from ml import backtest as backtest_mod
from ml import compute as compute_mod
from ml import model as model_mod
from ml import multi_horizon as multi_mod
from ml.backtest import walk_forward_backtest
from ml.compute import (
    GPUXGBClassifier,
    GPUXGBRegressor,
    accelerated_classifier,
    accelerated_regressor,
    gpu_backend_available,
    nvidia_gpu_available,
)
from ml.model import StockPredictor
from ml.multi_horizon import HorizonConfig, HorizonModel
from tests.conftest import FakeYFTicker, make_intraday_ohlcv, make_ohlcv


class DummyScaler:
    def fit_transform(self, X):
        return np.asarray(X, dtype=float)

    def transform(self, X):
        return np.asarray(X, dtype=float)


class DummyClassifier:
    def __init__(self, prediction=1):
        self.prediction = prediction
        self.fit_calls = 0
        self.named_estimators_ = {'rf': SimpleNamespace(feature_importances_=np.array([0.6, 0.4]))}

    def fit(self, X, y):
        self.fit_calls += 1
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict(self, X):
        return np.array([self.prediction] * len(X))

    def predict_proba(self, X):
        return np.tile(np.array([[0.2, 0.8]]), (len(X), 1))


class DummyRegressor:
    def __init__(self, value=0.02):
        self.value = value
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        return self

    def predict(self, X):
        return np.array([self.value] * len(X))


class FakeSplitter:
    def __init__(self, *splits):
        self._splits = list(splits)

    def split(self, X):
        return iter(self._splits)


class HistoryFactory:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, ticker):
        return FakeYFTicker(ticker, data=self.mapping[ticker])


def _frame_from_history(data: pd.DataFrame, columns=('f1', 'f2'), usable=120):
    frame = pd.DataFrame(index=data.index[:usable])
    for idx, column in enumerate(columns, start=1):
        frame[column] = np.linspace(0.0, float(idx), len(frame))
    return frame


def _targets_from_history(data: pd.DataFrame, usable=120):
    index = data.index[:usable]
    cls = pd.Series(np.resize(np.array([1, -1, 0]), len(index)), index=index)
    reg = pd.Series(np.linspace(-0.03, 0.03, len(index)), index=index)
    return cls, reg


def test_compute_gpu_detection_and_estimator_wrappers(monkeypatch):
    nvidia_gpu_available.cache_clear()
    monkeypatch.setattr(
        compute_mod.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=''),
    )
    assert nvidia_gpu_available() is False

    gpu_backend_available.cache_clear()
    monkeypatch.setattr(compute_mod, 'nvidia_gpu_available', lambda: True)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'xgboost':
            raise ImportError('missing xgboost')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    assert gpu_backend_available() is False

    builtins.__import__ = real_import
    gpu_backend_available.cache_clear()
    fake_xgb = types.SimpleNamespace(
        build_info=lambda: {'USE_CUDA': True},
        XGBRegressor=lambda **kwargs: SimpleNamespace(
            fit=lambda X, y: (_ for _ in ()).throw(RuntimeError('probe failed')),
            predict=lambda X: np.array([0.0]),
        ),
    )
    monkeypatch.setitem(sys.modules, 'xgboost', fake_xgb)
    monkeypatch.setattr(compute_mod, 'nvidia_gpu_available', lambda: True)
    assert gpu_backend_available() is False

    classifier = GPUXGBClassifier(fallback=None)
    classifier.model_ = SimpleNamespace(
        predict=lambda X: np.array([0, 1]),
        predict_proba=lambda X: np.array([[0.9, 0.1], [0.1, 0.9]]),
    )
    classifier.classes_ = np.array([-1, 1])
    classifier.using_fallback_ = False
    np.testing.assert_array_equal(classifier.predict([[1], [2]]), np.array([-1, 1]))
    np.testing.assert_array_equal(classifier.predict_proba([[1]]), np.array([[0.9, 0.1], [0.1, 0.9]]))

    regressor = GPUXGBRegressor(fallback=None)
    regressor.model_ = SimpleNamespace(predict=lambda X: np.array([3.5]))
    regressor.using_fallback_ = False
    np.testing.assert_array_equal(regressor.predict([[1]]), np.array([3.5]))

    monkeypatch.setattr(compute_mod, 'gpu_backend_available', lambda: False)
    assert accelerated_classifier('cpu-model') == 'cpu-model'
    assert accelerated_regressor('cpu-model') == 'cpu-model'
    monkeypatch.setattr(compute_mod, 'gpu_backend_available', lambda: True)
    assert isinstance(accelerated_classifier('cpu-model'), GPUXGBClassifier)
    assert isinstance(accelerated_regressor('cpu-model'), GPUXGBRegressor)


def test_stock_predictor_prepare_train_and_load(monkeypatch, tmp_path, isolated_env):
    short_history = make_ohlcv(n=100, seed=1)
    small_common = make_ohlcv(n=260, seed=2)
    good_history = make_ohlcv(n=260, seed=3)
    mapping = {'SHORT': short_history, 'SMALL': small_common, 'GOOD': good_history}
    monkeypatch.setattr(yfinance, 'Ticker', HistoryFactory(mapping))

    def fake_features(data, ticker='', **kwargs):
        if ticker == 'SMALL':
            return _frame_from_history(data, usable=90)
        return _frame_from_history(data, usable=120)

    monkeypatch.setattr(model_mod, 'compute_features', fake_features)
    monkeypatch.setattr(
        model_mod,
        'create_target',
        lambda data, horizon=10, threshold=0.0: _targets_from_history(data, 90 if len(data) == 260 and data.equals(small_common) else 120)[0],
    )
    monkeypatch.setattr(
        model_mod,
        'create_regression_target',
        lambda data, horizon=10: _targets_from_history(data, 90 if len(data) == 260 and data.equals(small_common) else 120)[1],
    )

    predictor = StockPredictor(horizon=2, threshold=0.01)
    X, y_cls, y_reg = predictor._prepare_data(['SHORT', 'SMALL', 'GOOD'])
    assert len(X) == len(y_cls) == len(y_reg) == 120
    assert list(X.columns) == ['f1', 'f2']

    train_X = pd.DataFrame({'f1': np.arange(60), 'f2': np.arange(60) * 2})
    y_cls = pd.Series(([1, -1] * 30))
    y_reg = pd.Series(np.linspace(-0.02, 0.02, 60))
    predictor.classifier = DummyClassifier(prediction=1)
    predictor.regressor = DummyRegressor(value=0.01)
    predictor.scaler = DummyScaler()
    monkeypatch.setattr(predictor, '_prepare_data', lambda tickers, timeline='5y', interval='1d': (train_X, y_cls, y_reg))
    monkeypatch.setattr(
        model_mod,
        'TimeSeriesSplit',
        lambda n_splits=5: FakeSplitter(
            (np.arange(40), np.arange(40, 50)),
            (np.arange(50), np.arange(50, 60)),
        ),
    )
    stats = predictor.train(['GOOD'])
    assert stats['class_distribution'] == {1: 30, -1: 30}
    assert len(stats['cv_scores']) == 2
    assert predictor.is_trained is True

    model_dir = tmp_path / 'models'
    model_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(model_mod, 'MODEL_DIR', model_dir)
    payload = {
        'classifier': DummyClassifier(),
        'regressor': DummyRegressor(),
        'scaler': DummyScaler(),
        'feature_names': ['f1', 'f2'],
        'training_stats': {'trained_at': '2026-01-01T00:00:00Z'},
        'horizon': 3,
        'threshold': 0.02,
        'model_name': 'renamed_model',
    }
    with open(model_dir / 'stock_predictor.pkl', 'wb') as handle:
        pickle.dump(payload, handle)
    reloaded = StockPredictor()
    reloaded.load()
    assert reloaded.model_name == 'renamed_model'
    assert reloaded.is_trained is True


def test_walk_forward_backtest_handles_no_data_and_predicts(monkeypatch):
    short = make_ohlcv(n=280, seed=4)
    mapping = {'SHORT': short}
    monkeypatch.setattr(backtest_mod.yfinance, 'Ticker', HistoryFactory(mapping))
    empty_result = walk_forward_backtest(['SHORT'], n_splits=2, min_train_size=100)
    assert empty_result.predictions == []
    assert empty_result.accuracy == 0.0

    full = make_ohlcv(n=620, seed=5)
    missing = make_ohlcv(n=620, seed=6)
    mapping = {'GOOD': full, 'MISS': missing}
    monkeypatch.setattr(backtest_mod.yfinance, 'Ticker', HistoryFactory(mapping))
    monkeypatch.setattr(
        backtest_mod,
        'compute_features',
        lambda data, ticker='', **kwargs: _frame_from_history(data, columns=('f1',) if ticker == 'MISS' else ('f1', 'f2'), usable=240),
    )
    monkeypatch.setattr(backtest_mod, 'create_target', lambda data, horizon=10, threshold=0.0: _targets_from_history(data, 240)[0])
    monkeypatch.setattr(backtest_mod, 'create_regression_target', lambda data, horizon=10: _targets_from_history(data, 240)[1])

    class FakePredictor:
        def __init__(self, horizon=10, threshold=0.0):
            self.horizon = horizon
            self.threshold = threshold
            self.classifier = DummyClassifier(prediction=1)
            self.regressor = DummyRegressor(value=0.02)
            self.scaler = DummyScaler()
            self.feature_names = []

    monkeypatch.setattr(backtest_mod, 'StockPredictor', FakePredictor)
    result = walk_forward_backtest(
        ['GOOD'],
        test_tickers=['GOOD', 'MISS'],
        total_history='5y',
        n_splits=2,
        min_train_size=100,
        horizon=5,
        confidence_floor=0.0,
    )
    assert result.predictions
    assert result.directional_accuracy >= 0.0
    assert all(item['ticker'] == 'GOOD' for item in result.predictions)


def test_horizon_model_prepare_train_predict_and_load(monkeypatch, tmp_path, isolated_env):
    intraday = make_intraday_ohlcv(n=220)
    daily = make_ohlcv(n=260, seed=8)
    history_map = {'SPY': daily, 'AAPL': daily, 'MSFT': daily, 'INTRA': intraday, 'REAL': make_intraday_ohlcv(n=600)}
    monkeypatch.setattr(multi_mod.yfinance, 'Ticker', HistoryFactory(history_map))
    monkeypatch.setattr(multi_mod, 'compute_features', lambda data, ticker='', **kwargs: _frame_from_history(data, usable=130))
    monkeypatch.setattr(multi_mod, 'compute_intraday_features', lambda data, ticker='': _frame_from_history(data, usable=130))
    monkeypatch.setattr(multi_mod, 'compute_realtime_features', lambda data, market_data: _frame_from_history(data, usable=130))
    monkeypatch.setattr(multi_mod, 'create_target', lambda data, horizon=1, threshold=0.0: _targets_from_history(data, 130)[0])
    monkeypatch.setattr(multi_mod, 'create_regression_target', lambda data, horizon=1: _targets_from_history(data, 130)[1])
    monkeypatch.setattr(multi_mod, 'create_intraday_targets', lambda data, horizon, threshold: _targets_from_history(data, 130))

    realtime_cfg = HorizonConfig('rt', 5, 0.001, '7d', '1m', 50, use_cross_asset=False, use_realtime_features=True, classifier_kind='extra_trees', target_price_mode='threshold')
    with pytest.raises(ValueError):
        HorizonModel(realtime_cfg).prepare_data(['REAL'], context=None)

    intraday_cfg = HorizonConfig('intraday_test', 5, 0.001, '60d', '5m', 50, use_cross_asset=False, use_intraday_features=True)
    intraday_model = HorizonModel(intraday_cfg)
    X_i, y_i, reg_i = intraday_model.prepare_data(['INTRA'])
    assert len(X_i) == len(y_i) == len(reg_i)

    train_cfg = HorizonConfig('short_test', 1, 0.0, '5y', '1d', 50, use_cross_asset=False, classifier_kind='ensemble', target_price_mode='regression')
    model = HorizonModel(train_cfg)
    model.classifier = DummyClassifier(prediction=1)
    model.regressor = DummyRegressor(value=0.01)
    model.scaler = DummyScaler()
    monkeypatch.setattr(model, 'prepare_data', lambda tickers, context=None: (pd.DataFrame({'f1': np.arange(80), 'f2': np.arange(80) * 0.5}), pd.Series(([1, -1] * 40)), pd.Series(np.linspace(-0.02, 0.02, 80))))
    monkeypatch.setattr(multi_mod, 'TimeSeriesSplit', lambda **kwargs: FakeSplitter((np.arange(50), np.arange(50, 60)), (np.arange(60), np.arange(60, 70))))
    stats = model.train(['AAPL'])
    assert stats['horizon'] == 'short_test'
    assert model.is_trained is True

    tight_cfg = HorizonConfig('tiny', 10, 0.0, '5y', '1d', 50, use_cross_asset=False)
    tiny_model = HorizonModel(tight_cfg)
    tiny_model.classifier = DummyClassifier()
    tiny_model.regressor = DummyRegressor()
    tiny_model.scaler = DummyScaler()
    monkeypatch.setattr(tiny_model, 'prepare_data', lambda tickers, context=None: (pd.DataFrame({'f1': np.arange(12)}), pd.Series([1, -1] * 6), pd.Series(np.linspace(-0.01, 0.01, 12))))
    with pytest.raises(ValueError):
        tiny_model.train(['AAPL'])

    model_dir = tmp_path / 'models'
    model_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(multi_mod, 'MODEL_DIR', model_dir)
    model.feature_names = ['f1', 'f2']
    model.is_trained = True
    prediction = model.predict_ticker('AAPL')
    assert prediction['direction'] == 'UP'
    model.save()
    loaded = HorizonModel(train_cfg)
    loaded.load()
    assert loaded.is_trained is True


class FakeWFHorizonModel:
    def __init__(self, config):
        self.config = config
        self.scaler = DummyScaler()
        self.classifier = DummyClassifier(prediction=1)
        self.regressor = DummyRegressor(value=0.01)
        self.feature_names = []


@pytest.mark.parametrize('name,target_price_mode', [('threshold_model', 'threshold'), ('regression_model', 'regression')])
def test_horizon_holdout_and_walk_forward(monkeypatch, name, target_price_mode):
    daily = make_ohlcv(n=220, seed=10)
    mapping = {'AAPL': daily, 'MSFT': daily, 'MISS': daily, 'SPY': daily}
    monkeypatch.setattr(multi_mod.yfinance, 'Ticker', HistoryFactory(mapping))
    monkeypatch.setattr(
        multi_mod,
        'compute_features',
        lambda data, ticker='', **kwargs: _frame_from_history(data, columns=('f1',) if ticker == 'MISS' else ('f1', 'f2'), usable=120),
    )
    monkeypatch.setattr(multi_mod, 'create_target', lambda data, horizon=1, threshold=0.0: _targets_from_history(data, 120)[0])
    monkeypatch.setattr(multi_mod, 'create_regression_target', lambda data, horizon=1: _targets_from_history(data, 120)[1])
    monkeypatch.setattr(multi_mod, 'HorizonModel', FakeWFHorizonModel)
    cfg = HorizonConfig(name, 1, 0.01, '5y', '1d', 50, use_cross_asset=False, classifier_kind='extra_trees' if target_price_mode == 'threshold' else 'ensemble', target_price_mode=target_price_mode)
    monkeypatch.setitem(multi_mod.HORIZONS, name, cfg)

    holdout = multi_mod.chronological_holdout_horizon(name, ['AAPL', 'MSFT'], test_days=2)
    assert holdout['total_predictions'] > 0
    assert holdout['feature_count'] == 2

    walked = multi_mod.walk_forward_horizon(name, ['AAPL'], test_tickers=['AAPL', 'MISS'], n_splits=2)
    assert walked['total_predictions'] > 0
    assert 'accuracy' in walked


def test_horizon_realtime_context_error(monkeypatch):
    monkeypatch.setattr(
        multi_mod.yfinance,
        'Ticker',
        lambda ticker: SimpleNamespace(history=lambda **kwargs: pd.DataFrame()),
    )
    cfg = HorizonConfig('rt_error', 5, 0.001, '7d', '1m', 50, use_cross_asset=False, use_realtime_features=True)
    monkeypatch.setitem(multi_mod.HORIZONS, 'rt_error', cfg)
    assert multi_mod.chronological_holdout_horizon('rt_error', ['AAPL']) == {'error': 'No SPY context data'}
    assert multi_mod.walk_forward_horizon('rt_error', ['AAPL']) == {'error': 'No SPY context data'}
