"""Shared fixtures for the backtester test suite.

Provides synthetic OHLCV data generators, yfinance mocks, and temporary
database/model infrastructure so every test runs network-free and
deterministically.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic OHLCV data
# ---------------------------------------------------------------------------

def make_ohlcv(
    n: int = 500,
    start: str = "2021-01-04",
    freq: str = "B",
    base_price: float = 100.0,
    seed: int = 42,
    *,
    volatility: float = 0.015,
    drift: float = 0.0003,
    volume_spikes: bool = True,
) -> pd.DataFrame:
    """Generate *realistic* synthetic OHLCV data.

    The close follows a geometric random walk so technical indicators
    (RSI, MACD, Bollinger, etc.) compute meaningfully.

    When volume_spikes=True, inject periodic volume shocks with
    correlated price drops/surges so strategies like VolumeShockReversal,
    ATRBreakout, and DonchianChannel trigger their buy/sell branches.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, periods=n, freq=freq)
    log_returns = rng.normal(drift, volatility, size=n)

    # Inject deliberate regime changes to trigger strategy branches:
    # Large drops every ~80 bars, surges every ~120 bars
    for i in range(n):
        if i % 80 == 40:
            log_returns[i] = -0.06  # sharp drop (triggers buy signals)
        elif i % 120 == 90:
            log_returns[i] = 0.05   # sharp surge (triggers sell signals)
        elif i % 60 == 30:
            log_returns[i] = 0.04   # moderate gain for profit-taking

    close = base_price * np.exp(np.cumsum(log_returns))
    high = close * (1 + rng.uniform(0.002, 0.03, n))
    low = close * (1 - rng.uniform(0.002, 0.03, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.randint(500_000, 5_000_000, n).astype(float)

    if volume_spikes:
        # Inject volume shocks coinciding with price moves
        for i in range(n):
            if i % 80 == 40 or i % 120 == 90:
                volume[i] = volume[i] * 4  # >2x shock

    # Make gaps between open and previous close for GapFade strategy
    for i in range(1, n):
        if i % 100 == 50:
            open_[i] = close[i-1] * 0.97  # gap down >1.5%

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index = df.index.tz_localize("America/New_York").tz_convert("UTC")
    df.index.name = None
    return df


def make_intraday_ohlcv(
    n: int = 600,
    start: str = "2026-08-28 09:30",
    freq: str = "1min",
    base_price: float = 150.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate intraday 1-minute synthetic data within a single session."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start=start, periods=n, freq=freq, tz="America/New_York")
    log_returns = rng.normal(0.0, 0.001, size=n)
    close = base_price * np.exp(np.cumsum(log_returns))
    high = close * (1 + rng.uniform(0.0005, 0.005, n))
    low = close * (1 - rng.uniform(0.0005, 0.005, n))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.randint(1000, 50_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates.tz_convert("UTC"),
    )


class FakeYFTicker:
    """Drop-in replacement for ``yfinance.Ticker``."""

    def __init__(self, symbol: str, data: pd.DataFrame | None = None):
        self.ticker = symbol
        self._data = data if data is not None else make_ohlcv()
        self.info = {
            "marketCap": 2_000_000_000_000,
            "trailingPE": 28.5,
            "forwardPE": 25.0,
            "priceToBook": 40.0,
            "revenueGrowth": 0.08,
            "earningsGrowth": 0.12,
            "debtToEquity": 150.0,
            "freeCashflow": 90_000_000_000,
        }
        self.options = ("2026-09-19",)
        self.news = [
            {
                "id": "fake-1",
                "content": {
                    "title": "Test stock rallies on earnings beat",
                    "canonicalUrl": {"url": "https://example.com/article"},
                    "pubDate": datetime.now(timezone.utc).isoformat(),
                },
            }
        ]

    def history(self, **kwargs) -> pd.DataFrame:
        period = kwargs.get("period", "1y")
        interval = kwargs.get("interval", "1d")
        if interval in ("1m", "5m", "15m", "30m", "60m", "1h"):
            return make_intraday_ohlcv()
        # Return appropriate length based on period
        length_map = {"5d": 5, "1mo": 22, "3mo": 66, "6mo": 130, "ytd": 200,
                      "1y": 252, "2y": 504, "5y": 1260, "10y": 2520, "max": 2520,
                      "60d": 42, "7d": 5}
        n = length_map.get(period, 252)
        n = min(n, len(self._data))
        return self._data.iloc[-n:].copy()

    def get_news(self, count=10):
        return self.news

    def option_chain(self, expiration):
        calls = pd.DataFrame({
            "volume": [100.0, 200.0],
            "impliedVolatility": [0.3, 0.35],
        })
        puts = pd.DataFrame({
            "volume": [80.0, 150.0],
            "impliedVolatility": [0.32, 0.38],
        })
        return SimpleNamespace(calls=calls, puts=puts)

    def get_earnings_dates(self, limit=12):
        dates = pd.date_range(end="2026-06-01", periods=4, freq="QE", tz="UTC")
        return pd.DataFrame(
            {
                "EPS Estimate": [1.5, 1.4, 1.3, 1.2],
                "Reported EPS": [1.6, 1.35, 1.4, 1.1],
                "Surprise(%)": [6.67, -3.57, 7.69, -8.33],
            },
            index=dates,
        )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ohlcv():
    """Return a 500-row synthetic OHLCV DataFrame."""
    return make_ohlcv()


@pytest.fixture()
def intraday_ohlcv():
    return make_intraday_ohlcv()


@pytest.fixture()
def tmp_model_dir(tmp_path):
    """Isolated directory for model artifacts and databases."""
    (tmp_path / "models").mkdir()
    return tmp_path


@pytest.fixture()
def isolated_env(tmp_path):
    """Set env vars to keep all DB/model writes in tmp_path."""
    db_path = tmp_path / "predictions.db"
    alt_db = tmp_path / "alt_data.db"
    cache_path = tmp_path / "cache.json"
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    env = {
        "PREDICTION_DB_PATH": str(db_path),
        "ALT_DATA_DB_PATH": str(alt_db),
        "WEBAPP_CACHE_PATH": str(cache_path),
        "ML_ACCELERATOR": "cpu",
    }
    with patch.dict(os.environ, env):
        yield tmp_path


@pytest.fixture()
def mock_yfinance():
    """Globally patch yfinance.Ticker to return FakeYFTicker."""
    shared_data = make_ohlcv(n=1300, seed=42)

    def factory(symbol):
        return FakeYFTicker(symbol, data=shared_data.copy())

    with patch("yfinance.Ticker", side_effect=factory) as mock:
        yield mock


@pytest.fixture(autouse=True)
def _reset_webapp_globals():
    """Ensure webapp module globals are clean for every test."""
    yield
    # Reset globals that tests may have set
    try:
        from webapp import app as webapp_mod
        webapp_mod._last_settlement_at = None
        webapp_mod._settlement_in_progress = False
        webapp_mod._prediction_refreshes.clear()
    except Exception:
        pass
    # Clear features cache
    try:
        from ml.features import _context_cache
        _context_cache.clear()
    except Exception:
        pass
