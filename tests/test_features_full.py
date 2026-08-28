import sys
import types
from unittest.mock import Mock, patch

import pandas as pd

from ml import features
from ml.features import (
    _add_alternative_data_features,
    _alternative_data_store,
    _download_context,
    compute_intraday_features,
)
from tests.conftest import make_intraday_ohlcv


class FakeAltStore:
    def get_feature_frame(self, ticker, index, lookback_days=30):
        return pd.DataFrame({'alt_x': [1.0] * len(index)}, index=index)


def test_alternative_data_store_import_is_cached(monkeypatch):
    fake_module = types.SimpleNamespace(AlternativeDataStore=lambda: FakeAltStore())
    monkeypatch.setitem(sys.modules, 'ml.alternative_data', fake_module)
    _alternative_data_store.cache_clear()
    store = _alternative_data_store()
    assert isinstance(store, FakeAltStore)
    assert _alternative_data_store() is store


def test_add_alternative_data_features_short_circuits_for_empty_ticker():
    frame = pd.DataFrame({'signal': [1.0]})
    assert _add_alternative_data_features(frame, '') is frame


def test_download_context_drops_none_results(monkeypatch):
    features._context_cache.clear()
    tickers = list(features.MARKET_CONTEXT_TICKERS) + list(features.SECTOR_ETFS)
    monkeypatch.setattr(
        features,
        'parallel_map',
        lambda function, items: [(ticker, None) for ticker in items],
    )
    assert _download_context('1y', '1d') == {}
    assert '1y_1d' in features._context_cache


def test_compute_intraday_features_includes_alternative_data(monkeypatch):
    data = make_intraday_ohlcv(n=120)
    fake_store = FakeAltStore()
    monkeypatch.setattr(features, '_add_alternative_data_features', lambda feat, ticker: pd.concat([feat, fake_store.get_feature_frame(ticker, feat.index)], axis=1))
    result = compute_intraday_features(data, ticker='AAPL', include_alternative_data=True)
    assert 'alt_x' in result.columns
    assert not result.empty
