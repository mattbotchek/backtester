import time

from ml.features import compute_features, compute_intraday_features
from ml.sentiment.combined_model import _build_synthetic_sentiment_history
from tests.conftest import make_intraday_ohlcv, make_ohlcv


def test_feature_engineering_performance_budget():
    daily = make_ohlcv(n=500)
    started = time.perf_counter()
    features = compute_features(daily, ticker='AAPL', include_cross_asset=False, include_alternative_data=False)
    elapsed = time.perf_counter() - started
    assert not features.empty
    assert elapsed < 5.0


def test_intraday_feature_engineering_performance_budget():
    intraday = make_intraday_ohlcv(n=600)
    started = time.perf_counter()
    features = compute_intraday_features(intraday, ticker='AAPL', include_alternative_data=False)
    elapsed = time.perf_counter() - started
    assert not features.empty
    assert elapsed < 5.0


def test_synthetic_sentiment_history_performance_budget():
    daily = make_ohlcv(n=500)
    started = time.perf_counter()
    result = _build_synthetic_sentiment_history(daily.index, ticker='AAPL', vix_data=daily['Close'])
    elapsed = time.perf_counter() - started
    assert not result.empty
    assert elapsed < 2.0
