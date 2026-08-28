import argparse
import json
import runpy
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yfinance

from ml import alternative_data as alt_mod
from ml import continuous_training as ct_mod
from ml import headless as headless_mod
from ml import optimize_loop as opt_mod
from ml import prediction_store as ps_mod
from ml.continuous_training import ContinuousTrainingService
from ml.prediction_store import PredictionStore
from tests.conftest import FakeYFTicker, make_ohlcv

UTC = timezone.utc


class HistoryFactory:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, ticker):
        return FakeYFTicker(ticker, data=self.mapping[ticker])


class DummyModel:
    def __init__(self, payload=None):
        self.payload = payload or {}
        self._training_stats = {'trained_at': '2026-01-01T00:00:00Z'}

    def load(self, *args, **kwargs):
        return None

    def predict(self, ticker):
        if ticker == 'FAIL':
            raise RuntimeError('predict failed')
        return {'ticker': ticker}

    def predict_ticker(self, ticker):
        if ticker == 'FAIL':
            raise RuntimeError('predict failed')
        return {'ticker': ticker}

    def train(self, tickers, timeline='5y'):
        return {'n_samples': len(tickers), 'cv_accuracy_mean': 0.8}

    def save(self):
        return None


class StopEvent:
    def __init__(self):
        self.calls = 0

    def is_set(self):
        return self.calls > 0

    def wait(self, seconds):
        self.calls += 1
        return True


def test_prediction_store_time_helpers_and_settlement_cache(monkeypatch, isolated_env):
    friday = datetime(2026, 8, 28, 20, 0, tzinfo=UTC)
    assert ps_mod.to_utc(datetime(2026, 8, 28, 20, 0)).tzinfo is UTC
    assert ps_mod.iso_utc(datetime(2026, 8, 28, 20, 0)) == '2026-08-28T20:00:00Z'
    assert ps_mod.prediction_target_time(friday, 1, '1wk').weekday() == 4
    assert ps_mod.prediction_target_time(friday, 2, '2d') > friday
    assert ps_mod.prediction_target_time(friday, 1, 'custom') == friday + timedelta(days=1)
    assert ps_mod.directional_hit('FLAT', 100.0, 100.2) is True
    assert ps_mod.directional_hit('DOWN', 98.0, 97.0) is True
    with pytest.raises(ValueError):
        ps_mod._advance_market_bars(pd.Timestamp(friday), pd.Timedelta(minutes=5), 0)
    with pytest.raises(ValueError):
        ps_mod._advance_market_sessions(pd.Timestamp(friday), 0)
    with pytest.raises(ValueError):
        ps_mod.prediction_target_time(datetime(2026, 8, 28, 18, 0, tzinfo=UTC), 30, '5m')
    assert ps_mod._bar_completion_time('2026-08-24T13:30:00Z', '1wk').weekday() == 4
    assert ps_mod._bar_completion_time('2026-08-28T13:30:00Z', '1d').hour >= 18

    store = PredictionStore()
    target = ps_mod.prediction_target_time(friday, 1, '1d')
    first = store.record_prediction(
        model_name='m1', family='test', ticker='AAPL', input_at=friday,
        target_at=target, interval='1d', horizon=1, base_price=100,
        predicted_return_pct=1.0, direction='UP', confidence=0.7,
    )
    second = store.record_prediction(
        model_name='m2', family='test', ticker='AAPL', input_at=friday,
        target_at=target, interval='1d', horizon=1, base_price=100,
        predicted_return_pct=1.0, direction='UP', confidence=0.7,
    )
    calls = {'count': 0}
    monkeypatch.setattr(
        ps_mod,
        '_fetch_market_prices',
        lambda ticker, target_at, interval: calls.__setitem__('count', calls['count'] + 1) or [(target_at, 101.0)],
    )
    result = store.settle_due(now=target)
    assert result['settled'] == 2
    assert calls['count'] == 1
    settled = store.settle_prediction(first['id'], 101.0, target)
    assert settled['status'] == 'settled'
    assert store.settle_prediction(first['id'], 101.0, target)['status'] == 'settled'

    pending = store.record_prediction(
        model_name='m3', family='test', ticker='MSFT', input_at=friday,
        target_at=target, interval='1d', horizon=1, base_price=100,
        predicted_return_pct=1.0, direction='UP', confidence=0.7,
    )
    unavailable = store.settle_due(now=target, price_fetcher=lambda *args: [])
    assert unavailable['unavailable'] >= 1
    errored = store.settle_due(now=target, price_fetcher=lambda *args: (_ for _ in ()).throw(RuntimeError('fetch')))
    assert any('fetch' in item for item in errored['errors'])

    daily = make_ohlcv(n=5)
    intraday = make_ohlcv(n=5)
    monkeypatch.setattr(
        yfinance,
        'Ticker',
        lambda ticker: SimpleNamespace(history=lambda **kwargs: daily if kwargs.get('interval') == '1d' else intraday),
    )
    assert ps_mod._fetch_market_prices('AAPL', target, '1d')
    assert ps_mod._fetch_market_prices('AAPL', target, '1m')


def test_continuous_training_service_paths(monkeypatch, tmp_path, isolated_env):
    model_dir = tmp_path / 'models'
    model_dir.mkdir(exist_ok=True)
    state_path = tmp_path / 'state.json'
    monkeypatch.setattr(ct_mod, 'MODEL_DIR', model_dir)
    custom_horizons = {'mini': ct_mod.HORIZONS['short_1d']}
    monkeypatch.setattr(ct_mod, 'HORIZONS', custom_horizons)

    for filename in ['stock_predictor.pkl', 'combined_model.pkl', 'model_short_1d.pkl']:
        (model_dir / filename).write_text('x', encoding='utf-8')

    monkeypatch.setattr(ct_mod, 'StockPredictor', DummyModel)
    monkeypatch.setattr(ct_mod, 'HorizonModel', lambda config: DummyModel())
    monkeypatch.setattr(ct_mod, 'train_all_horizons', lambda tickers: {'mini': {'score': 1.0}})
    fake_combined_module = SimpleNamespace(CombinedPredictor=DummyModel)
    monkeypatch.setitem(__import__('sys').modules, 'ml.sentiment.combined_model', fake_combined_module)

    collector = SimpleNamespace(collect_all=lambda tickers: {'inserted': len(tickers)})
    monkeypatch.setitem(__import__('sys').modules, 'ml.alternative_data', SimpleNamespace(AlternativeDataCollector=lambda: collector))

    service = ContinuousTrainingService(['AAPL', 'FAIL'], state_path=state_path)
    assert service.collect_alternative_data()['inserted'] == 2
    prediction_result = service._predict_with_model(DummyModel(), 'default')
    assert prediction_result['successes'] == 1
    assert prediction_result['failures']

    generated = service.generate_predictions()
    assert 'stock_predictor' in generated and 'combined_model' in generated

    service.store.register_model('stock_predictor', 'StockPredictor', 1, '1d')
    service.store.record_prediction(
        model_name='stock_predictor', family='StockPredictor', ticker='AAPL',
        input_at='2026-01-01T00:00:00Z', target_at='2026-01-02T00:00:00Z',
        interval='1d', horizon=1, base_price=100.0, predicted_return_pct=1.0,
        direction='UP', confidence=0.7,
    )
    service.store.settle_due(now='2026-01-03T00:00:00Z', price_fetcher=lambda *args: [('2026-01-02T00:00:00Z', 101.0)])
    evaluation = service.evaluate_forward_accuracy()
    assert evaluation['models']

    monkeypatch.setattr(service, 'retrain_all_models', lambda: {'models': {'stock_predictor': {'ok': True}}, 'failures': []})
    cycle = service.run_cycle(force_retrain=True)
    assert 'training' in cycle and 'predictions' in cycle and 'collection' in cycle
    assert state_path.exists()

    bad_state = tmp_path / 'bad.json'
    bad_state.write_text('{bad json', encoding='utf-8')
    with pytest.raises(RuntimeError):
        ContinuousTrainingService(['AAPL'], state_path=bad_state)


def test_continuous_training_run_forever_and_retraining_due(monkeypatch, tmp_path, isolated_env):
    service = ContinuousTrainingService(['AAPL'], state_path=tmp_path / 'state.json', stop_event=StopEvent())
    monkeypatch.setattr(service, 'run_cycle', lambda: {'forward_evaluation': {'best_model': 'm', 'best_forward_accuracy': 0.8, 'target_achieved': True}})
    service.run_forever()
    assert service.stop_event.calls == 1
    assert service.retraining_due() is True


def test_headless_runner_and_main(monkeypatch, tmp_path, capsys):
    saved_handlers = {}
    csv_path = tmp_path / 'headless.csv'

    class FakePrediction:
        def __init__(self, ticker):
            self.timestamp = '2026-01-01T00:00:00Z'
            self.model_name = 'stock_predictor'
            self.ticker = ticker
            self.input_at = '2026-01-01T00:00:00Z'
            self.predicted_for = '2026-01-02T00:00:00Z'
            self.base_price = 100.0
            self.predicted_price = 101.0
            self.direction = 'UP'
            self.predicted_return_pct = 1.0
            self.confidence = 0.8
            self.forward_accuracy = None
            self.settled_predictions = 0
            self.last_trained_at = '2026-01-01T00:00:00Z'
            self.outliers = []

        def __repr__(self):
            return f'{self.ticker} prediction'

    class HeadlessPredictor(DummyModel):
        def load(self, model_name='stock_predictor'):
            return None

        def predict(self, ticker):
            return FakePrediction(ticker)

    monkeypatch.setattr(headless_mod, 'StockPredictor', HeadlessPredictor)
    monkeypatch.setattr(headless_mod.signal, 'signal', lambda sig, handler: saved_handlers.setdefault(sig, handler))
    monkeypatch.setattr(headless_mod.PredictionStore, 'settle_due', lambda self: {'settled': 1})

    def fake_sleep(seconds):
        saved_handlers[headless_mod.signal.SIGINT](None, None)

    monkeypatch.setattr(headless_mod.time, 'sleep', fake_sleep)
    headless_mod.run_headless(tickers=['AAPL'], interval_seconds=1, output_file=str(csv_path))
    assert csv_path.exists()
    assert 'Recorded At' in csv_path.read_text(encoding='utf-8')
    assert 'Headless runner stopped.' in capsys.readouterr().out

    class MissingPredictor(HeadlessPredictor):
        def load(self, model_name='stock_predictor'):
            raise FileNotFoundError('missing')

    monkeypatch.setattr(headless_mod, 'StockPredictor', MissingPredictor)
    with pytest.raises(SystemExit):
        headless_mod.run_headless(tickers=['AAPL'], interval_seconds=1)

    called = {}
    monkeypatch.setattr(headless_mod, 'run_headless', lambda **kwargs: called.update(kwargs))
    monkeypatch.setattr(headless_mod.argparse.ArgumentParser, 'parse_args', lambda self: argparse.Namespace(model='m', tickers=['AAPL'], interval=5, output='out.csv'))
    headless_mod.main()
    assert called['model_name'] == 'm'


def test_optimize_loop_prepare_features_and_run_trial(monkeypatch):
    data = make_ohlcv(n=260, seed=12)
    ticker_data = {'AAPL': data, 'MSFT': data}
    monkeypatch.setattr(opt_mod, 'compute_features', lambda data, ticker='', **kwargs: pd.DataFrame({'f1': np.arange(120), 'f2': np.arange(120) * 2}, index=data.index[:120]))
    monkeypatch.setattr(opt_mod, 'create_target', lambda data, horizon=1, threshold=0.0: pd.Series(np.resize(np.array([1, -1, 0]), 120), index=data.index[:120]))
    X, y = opt_mod.prepare_features(ticker_data, {}, horizon=1, threshold=0.0, smooth=0)
    assert len(X) == len(y)

    cfg = opt_mod.TrialConfig(name='trial', horizon=1, threshold=0.0)

    class FakeVote:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([1] * len(X))

    monkeypatch.setattr(opt_mod, 'VotingClassifier', lambda *args, **kwargs: FakeVote())
    monkeypatch.setattr(opt_mod, 'HistGradientBoostingClassifier', lambda *args, **kwargs: FakeVote())
    monkeypatch.setattr(opt_mod, 'TimeSeriesSplit', lambda n_splits=5: SimpleNamespace(split=lambda X: iter([(np.arange(80), np.arange(80, 100)), (np.arange(100), np.arange(100, 120))])))
    monkeypatch.setattr(opt_mod, 'StandardScaler', lambda: SimpleNamespace(fit_transform=lambda X: X.to_numpy(), transform=lambda X: X.to_numpy()))
    result = opt_mod.run_trial(cfg, {'AAPL': data}, {'MSFT': data}, {})
    assert result['heldout_accuracy'] >= 0.0

    monkeypatch.setattr(opt_mod, 'compute_features', lambda data, ticker='', **kwargs: pd.DataFrame({'f1': np.arange(120)}, index=data.index[:120]) if ticker == 'MSFT' else pd.DataFrame({'f1': np.arange(120), 'f2': np.arange(120) * 2, 'f3': np.arange(120) * 3}, index=data.index[:120]))
    mismatch = opt_mod.run_trial(cfg, {'AAPL': data}, {'MSFT': data}, {})
    assert mismatch['error'] == 'Feature mismatch'
    opt_mod._print_summary([{'trial': 'x', 'cv_accuracy': 0.5, 'heldout_accuracy': 0.4}], None)


def test_alternative_data_helpers_and_collectors(monkeypatch, tmp_path):
    assert alt_mod._as_utc(1_725_000_000_000).tzinfo is UTC
    fallback = datetime(2026, 1, 1, tzinfo=UTC)
    assert alt_mod._as_utc('', fallback=fallback) == fallback
    assert alt_mod._as_utc('2026-01-01', end_of_day=True).hour == 23
    monkeypatch.setenv('ALT_TEST_ENDPOINTS', json.dumps([{'name': 'alpha', 'url': 'https://example.test'}]))
    assert alt_mod._endpoint_config('ALT_TEST_ENDPOINTS') == [('alpha', 'https://example.test')]

    store = alt_mod.AlternativeDataStore(tmp_path / 'alt.db')
    normalized = store._normalized_event({
        'category': 'news', 'source': 'unit', 'event_time': '2026-01-01T00:00:00Z',
        'available_at': '2026-01-01T00:00:00Z', 'collected_at': '2026-01-01T00:00:00Z',
        'payload': {'title': 'ok'}, 'external_id': None,
    })
    assert normalized['ticker'] is None
    with pytest.raises(ValueError):
        store._normalized_event({'category': 'news', 'source': 'x', 'event_time': '2026-01-01T00:00:00Z', 'available_at': '2026-01-01T00:00:00Z', 'collected_at': '2026-01-01T00:00:00Z', 'payload': 'bad'})

    session = SimpleNamespace(
        get=lambda url, **kwargs: SimpleNamespace(
            json=lambda: {'events': [{'event_time': '2026-01-01T00:00:00Z', 'category': 'weird', 'payload': {}, 'id': '1'}]},
            content=b'<rss><channel><item><title>AAPL rally</title><link>x</link></item></channel></rss>',
            raise_for_status=lambda: None,
            headers={'content-type': 'application/json'},
            status_code=200,
        )
    )
    collector = alt_mod.AlternativeDataCollector(store, session=session, now=lambda: fallback, enable_usgs=False, enable_eonet=False, enable_gdelt=False, news_feeds=[('rss', 'https://rss')], custom_json_feeds=[('custom', 'https://custom')])
    monkeypatch.setattr(alt_mod.feedparser, 'parse', lambda content: SimpleNamespace(entries=[{'title': 'AAPL rally', 'summary': 'growth', 'link': 'x'}], bozo=False))
    fetched, inserted = collector.collect_rss_feed('rss', 'https://rss', category='news')
    assert fetched == inserted == 1
    fetched, inserted = collector.collect_custom_json('custom', 'https://custom')
    assert fetched == inserted == 1


def test_optimize_loop_main_rounds(monkeypatch):
    cfg1 = opt_mod.TrialConfig(name='cfg1', horizon=5, threshold=0.01)
    cfg2 = opt_mod.TrialConfig(name='cfg2', horizon=5, threshold=0.02)
    monkeypatch.setattr(opt_mod, 'download_all_data', lambda tickers: {'AAPL': make_ohlcv(n=320, seed=13)})
    monkeypatch.setattr(opt_mod, '_download_context', lambda *args: {})
    monkeypatch.setattr(opt_mod, 'generate_trials', lambda horizon: [cfg1, cfg2])

    calls = []
    def fake_run_trial(cfg, train_data, test_data, context, n_splits=5):
        calls.append(cfg.name)
        if len(calls) == 1:
            return {'trial': cfg.name, 'heldout_accuracy': 0.6, 'cv_accuracy': 0.55, 'cv_std': 0.01, 'config': cfg}
        return {'trial': cfg.name, 'heldout_accuracy': 0.72, 'cv_accuracy': 0.6, 'cv_std': 0.02, 'config': cfg}

    monkeypatch.setattr(opt_mod, 'run_trial', fake_run_trial)
    results = opt_mod.run_optimization_loop(horizon=5, target_accuracy=0.7, max_rounds=2)
    assert any(result.get('heldout_accuracy') == 0.72 for result in results)

    monkeypatch.setattr(opt_mod, 'generate_trials', lambda horizon: [cfg1])
    monkeypatch.setattr(opt_mod, 'run_trial', lambda cfg, train_data, test_data, context, n_splits=5: {'trial': cfg.name, 'error': 'failed'})
    no_success = opt_mod.run_optimization_loop(horizon=5, target_accuracy=0.9, max_rounds=1)
    assert no_success == [{'trial': 'cfg1', 'error': 'failed'}]
