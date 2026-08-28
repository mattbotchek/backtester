import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml.sentiment import combined_model as cm
from ml.sentiment import scrapers as sc
from tests.conftest import FakeYFTicker, make_ohlcv


class DummyScaler:
    def fit_transform(self, X):
        return np.asarray(X, dtype=float)

    def transform(self, X):
        return np.asarray(X, dtype=float)


class DummyVote:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([1] * len(X))

    def predict_proba(self, X):
        return np.tile(np.array([[0.2, 0.8]]), (len(X), 1))


class DummyReg:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([0.02] * len(X))


def _patch_vader(monkeypatch):
    fake = SimpleNamespace(SentimentIntensityAnalyzer=lambda: SimpleNamespace(polarity_scores=lambda text: {'compound': 0.4}))
    monkeypatch.setitem(sys.modules, 'vaderSentiment', SimpleNamespace(vaderSentiment=fake))
    monkeypatch.setitem(sys.modules, 'vaderSentiment.vaderSentiment', fake)


def test_scrapers_cover_duplicate_fallback_and_invalid_paths(monkeypatch, tmp_path):
    db_path = tmp_path / 'sentiment.db'
    monkeypatch.setattr(sc, 'DB_PATH', db_path)
    _patch_vader(monkeypatch)

    first_entry = {'title': 'Apple rally', 'summary': 'AAPL surged', 'link': 'https://example/1', 'published': '2026-01-01'}
    monkeypatch.setattr(sc.feedparser, 'parse', lambda url: SimpleNamespace(entries=[first_entry]))
    first = sc.scrape_rss_news()
    second = sc.scrape_rss_news()
    assert len(first) == 1
    assert second == []

    class NewsTicker:
        def __init__(self):
            self.news = [{'title': 'MSFT gain', 'url': 'https://example/2', 'publisher': 'yf', 'providerPublishTime': 1_700_000_000}]

    monkeypatch.setattr(sc.yfinance, 'Ticker', lambda ticker: NewsTicker())
    articles = sc.scrape_yfinance_news(['MSFT'])
    assert articles[0]['tickers'] == ['MSFT']

    monkeypatch.setattr(sc.requests, 'get', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('cnn down')))
    monkeypatch.setattr(sc.yfinance, 'Ticker', lambda ticker: FakeYFTicker('^VIX', data=make_ohlcv(n=2)))
    fg = sc.scrape_fear_greed()
    assert fg is not None

    monkeypatch.setattr(
        sc.requests,
        'get',
        lambda url, timeout=10: SimpleNamespace(status_code=200, text='DATE,VALUE\n2026-01-01,not-a-number')
    )
    assert sc.scrape_fred_indicators() == {}

    monkeypatch.setattr(
        sc.requests,
        'get',
        lambda *args, **kwargs: SimpleNamespace(status_code=200, headers={'content-type': 'text/html'}, json=lambda: {'ignored': True}),
    )
    assert sc.scrape_sec_insider(['AAPL']) == []
    summary = sc.get_aggregated_sentiment()
    assert 'fear_greed_index' in summary


def test_combined_predictor_prepare_predict_walk_forward_and_accuracy_loop(monkeypatch, tmp_path, isolated_env):
    history = make_ohlcv(n=320, seed=21)
    monkeypatch.setattr(cm, '_download_context', lambda *args: {})
    monkeypatch.setattr(cm, 'compute_features', lambda data, ticker='', **kwargs: pd.DataFrame({'f1': np.arange(150), 'f2': np.arange(150) * 2}, index=data.index[:150]))
    monkeypatch.setattr(cm, '_build_synthetic_sentiment_history', lambda dates, ticker='', vix_data=None: pd.DataFrame({'sentiment_placeholder': np.zeros(len(dates)), 'fg_proxy': np.ones(len(dates))}, index=dates))
    monkeypatch.setattr(cm, '_build_macro_features', lambda dates: pd.DataFrame({'macro_rate_env_level': np.ones(len(dates))}, index=dates))
    monkeypatch.setattr(cm, 'create_target', lambda data, horizon=5, threshold=0.0: pd.Series(np.resize(np.array([1, -1, 0]), 150), index=data.index[:150]))
    monkeypatch.setattr(cm, 'create_regression_target', lambda data, horizon=5: pd.Series(np.linspace(-0.03, 0.03, 150), index=data.index[:150]))
    monkeypatch.setattr(cm.yfinance, 'Ticker', lambda ticker: FakeYFTicker(ticker, data=history))

    predictor = cm.CombinedPredictor(horizon=5, threshold=0.0, use_sentiment=True, use_macro=True)
    predictor.classifier = DummyVote()
    predictor.regressor = DummyReg()
    predictor.scaler = DummyScaler()
    X, y, y_reg, dates = predictor.prepare_data(['AAPL'])
    assert 'fg_proxy' in X.columns and 'macro_rate_env_level' in X.columns

    predictor.feature_names = list(X.columns)
    predictor.is_trained = True
    predictor._training_stats = {'trained_at': '2026-01-01T00:00:00Z'}
    pred = predictor.predict('AAPL')
    assert pred['direction'] == 'UP'

    monkeypatch.setattr(cm, 'VotingClassifier', lambda *args, **kwargs: DummyVote())
    monkeypatch.setattr(cm, 'StandardScaler', lambda: DummyScaler())
    walk_X = pd.DataFrame({'f1': np.arange(700), 'f2': np.arange(700) * 2.0})
    walk_y = pd.Series(np.resize(np.array([1, -1, 0]), 700))
    walk_dates = pd.Series(pd.date_range('2024-01-01', periods=700, freq='B', tz='UTC'))
    monkeypatch.setattr(predictor, 'prepare_data', lambda tickers, context: (walk_X, walk_y, walk_y.astype(float) / 100, walk_dates))
    result = predictor.walk_forward_backtest(['AAPL'], train_window=504, step_size=63)
    assert result['overall_accuracy'] >= 0.0

    model_dir = tmp_path / 'models'
    model_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(cm, 'MODEL_DIR', model_dir)
    predictor.save('combined_test')
    loaded = cm.CombinedPredictor()
    loaded.load('combined_test')
    assert loaded.model_name == 'combined_test'

    class LoopPredictor:
        def __init__(self, horizon=5, threshold=0.0, use_sentiment=True, use_macro=True):
            self.horizon = horizon
            self.threshold = threshold
            self.use_sentiment = use_sentiment
            self.use_macro = use_macro

        def walk_forward_backtest(self, tickers, train_window=504, step_size=63):
            if train_window == 504 and step_size == 63:
                return {'overall_accuracy': 0.7, 'confidence_filtered': {'conf_0.6': {'accuracy': 0.75, 'count': 60, 'pct': 50}}}
            return {'error': 'skip'}

        def train(self, tickers):
            return {'ok': True}

        def save(self, name):
            return None

    monkeypatch.setattr(cm, 'CombinedPredictor', LoopPredictor)
    best = cm.accuracy_loop(['AAPL'], target_accuracy=0.65, max_iterations=2)
    assert best['horizon'] == 5
