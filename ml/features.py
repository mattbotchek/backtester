"""Feature engineering — computes technical indicators for ML input.

Transforms raw OHLCV data into a feature matrix. All features are
look-back only (no future data leakage).

Includes:
  - Standard technical indicators (RSI, MACD, Bollinger, etc.)
  - Cross-asset context (SPY, VIX, sector ETFs, treasuries)
  - Multi-timeframe features
  - Market regime detection
"""

import logging
import threading
import warnings
from functools import lru_cache

# Suppress pandas fragmentation warnings — expected with incremental feature builds
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

import numpy as np
import pandas as pd
import yfinance

from .compute import parallel_map

logger = logging.getLogger(__name__)

# Cross-asset tickers for market context
MARKET_CONTEXT_TICKERS = {
    "SPY": "sp500",          # S&P 500
    "^VIX": "vix",           # Volatility index
    "QQQ": "nasdaq",         # Nasdaq 100
    "IWM": "russell2000",    # Russell 2000 (small cap)
    "TLT": "treasury_20y",   # 20-year treasury bonds
    "GLD": "gold",           # Gold
    "UUP": "dollar",         # US Dollar index
    "HYG": "high_yield",     # High yield corporate bonds
    "^TNX": "treasury_10y",   # 10-year Treasury yield
    "^IRX": "treasury_3m",    # 3-month Treasury yield
    "RSP": "equal_weight",    # Equal-weight S&P breadth proxy
    "CL=F": "crude_oil",      # Crude oil futures
    "HG=F": "copper",         # Copper growth proxy
    "BTC-USD": "bitcoin",     # Crypto risk appetite
}

SECTOR_ETFS = {
    "XLK": "tech", "XLF": "finance", "XLE": "energy",
    "XLV": "health", "XLY": "consumer_disc", "XLP": "consumer_staple",
    "XLI": "industrial", "XLU": "utilities", "XLRE": "real_estate",
    "XLB": "materials", "XLC": "communications",
}

# Map tickers to their sector ETF for relative strength
TICKER_SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLC", "NVDA": "XLK",
    "META": "XLC", "AMZN": "XLY", "TSLA": "XLY", "AMD": "XLK",
    "INTC": "XLK", "CRM": "XLK", "ORCL": "XLK", "ADBE": "XLK",
    "CSCO": "XLK", "AVGO": "XLK", "TXN": "XLK",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "V": "XLK",
    "MA": "XLK", "BRK-B": "XLF", "WFC": "XLF", "C": "XLF",
    "AXP": "XLF", "BLK": "XLF",
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABBV": "XLV",
    "MRK": "XLV", "LLY": "XLV", "TMO": "XLV", "ABT": "XLV",
    "BMY": "XLV", "AMGN": "XLV",
    "KO": "XLP", "PEP": "XLP", "PG": "XLP", "WMT": "XLP",
    "COST": "XLP", "MCD": "XLY", "NKE": "XLY", "SBUX": "XLY",
    "HD": "XLY", "LOW": "XLY",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    "BA": "XLI", "CAT": "XLI", "UNP": "XLI", "HON": "XLI",
    "GE": "XLI", "MMM": "XLI", "RTX": "XLI",
    "AMT": "XLRE", "PLD": "XLRE",
    "NEE": "XLU", "DUK": "XLU",
    "LIN": "XLB", "APD": "XLB",
}


_context_cache: dict[str, dict[str, pd.DataFrame]] = {}
_context_cache_lock = threading.Lock()


@lru_cache(maxsize=1)
def _alternative_data_store():
    from .alternative_data import AlternativeDataStore

    return AlternativeDataStore()


def _add_alternative_data_features(
    feat: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    if not ticker:
        return feat
    alternative = _alternative_data_store().get_feature_frame(
        ticker,
        feat.index,
        lookback_days=30,
    )
    return pd.concat([feat, alternative], axis=1)


def _download_context(period: str = "5y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Download market context data (cached per session)."""
    cache_key = f"{period}_{interval}"
    with _context_cache_lock:
        if cache_key in _context_cache:
            return _context_cache[cache_key]

    all_tickers = list(MARKET_CONTEXT_TICKERS.keys()) + list(SECTOR_ETFS.keys())

    def download(ticker: str) -> tuple[str, pd.DataFrame | None]:
        try:
            data = yfinance.Ticker(ticker).history(period=period, interval=interval)
            if not data.empty:
                return ticker, data
        except Exception as exc:
            logger.debug("Market context download failed for %s: %s", ticker, exc)
        return ticker, None

    context = {
        ticker: data
        for ticker, data in parallel_map(download, all_tickers)
        if data is not None
    }
    with _context_cache_lock:
        _context_cache[cache_key] = context
    return context


def _add_cross_asset_features(
    feat: pd.DataFrame,
    context: dict[str, pd.DataFrame],
    ticker: str = "",
) -> pd.DataFrame:
    """Add cross-asset and market context features."""
    for ctx_ticker, prefix in MARKET_CONTEXT_TICKERS.items():
        if ctx_ticker not in context:
            continue
        ctx = context[ctx_ticker]
        # Align to feat index
        ctx_close = ctx["Close"].reindex(feat.index, method="ffill")
        if ctx_close.isna().all():
            continue

        feat[f"{prefix}_return_1d"] = ctx_close.pct_change(1)
        feat[f"{prefix}_return_5d"] = ctx_close.pct_change(5)
        feat[f"{prefix}_return_20d"] = ctx_close.pct_change(20)
        feat[f"{prefix}_sma_ratio"] = ctx_close / ctx_close.rolling(20).mean() - 1

        # VIX level is directly useful
        if prefix == "vix":
            feat["vix_level"] = ctx_close
            feat["vix_sma_10"] = ctx_close.rolling(10).mean()
            feat["vix_change"] = ctx_close.pct_change(1)

    # Sector relative strength
    sector_etf = TICKER_SECTOR_MAP.get(ticker)
    if sector_etf and sector_etf in context:
        sector_close = context[sector_etf]["Close"].reindex(feat.index, method="ffill")
        if not sector_close.isna().all() and "Close" in feat.columns:
            stock_ret = feat["Close"].pct_change(20)
            sector_ret = sector_close.pct_change(20)
            feat["sector_relative_strength"] = stock_ret - sector_ret
            feat["sector_return_1d"] = sector_close.pct_change(1)
            feat["sector_return_5d"] = sector_close.pct_change(5)

    # Market breadth proxy: tech vs defensive
    if "XLK" in context and "XLU" in context:
        tech = context["XLK"]["Close"].reindex(feat.index, method="ffill")
        util = context["XLU"]["Close"].reindex(feat.index, method="ffill")
        if not tech.isna().all() and not util.isna().all():
            feat["risk_on_off"] = tech.pct_change(20) - util.pct_change(20)

    if "RSP" in context and "SPY" in context:
        equal_weight = context["RSP"]["Close"].reindex(feat.index, method="ffill")
        cap_weight = context["SPY"]["Close"].reindex(feat.index, method="ffill")
        feat["market_breadth_5d"] = (
            equal_weight.pct_change(5) - cap_weight.pct_change(5)
        )
        feat["market_breadth_20d"] = (
            equal_weight.pct_change(20) - cap_weight.pct_change(20)
        )

    if "XLY" in context and "XLP" in context:
        discretionary = context["XLY"]["Close"].reindex(feat.index, method="ffill")
        staples = context["XLP"]["Close"].reindex(feat.index, method="ffill")
        feat["consumer_risk_appetite"] = (
            discretionary.pct_change(20) - staples.pct_change(20)
        )

    # Credit spread proxy: HYG vs TLT
    if "HYG" in context and "TLT" in context:
        hyg = context["HYG"]["Close"].reindex(feat.index, method="ffill")
        tlt = context["TLT"]["Close"].reindex(feat.index, method="ffill")
        if not hyg.isna().all() and not tlt.isna().all():
            feat["credit_spread_proxy"] = hyg.pct_change(20) - tlt.pct_change(20)

    if "^TNX" in context and "^IRX" in context:
        ten_year = context["^TNX"]["Close"].reindex(feat.index, method="ffill")
        three_month = context["^IRX"]["Close"].reindex(feat.index, method="ffill")
        feat["yield_curve_10y_3m"] = ten_year - three_month
        feat["yield_curve_change_20d"] = feat["yield_curve_10y_3m"].diff(20)

    if "SPY" in context and "Close" in feat.columns:
        market_return = (
            context["SPY"]["Close"]
            .reindex(feat.index, method="ffill")
            .pct_change()
        )
        stock_return = feat["Close"].pct_change()
        market_variance = market_return.rolling(60).var()
        feat["market_beta_60d"] = (
            stock_return.rolling(60).cov(market_return) / market_variance
        )
        feat["market_corr_60d"] = stock_return.rolling(60).corr(market_return)
        feat["market_relative_20d"] = (
            feat["Close"].pct_change(20)
            - context["SPY"]["Close"]
            .reindex(feat.index, method="ffill")
            .pct_change(20)
        )

    return feat


def compute_features(
    df: pd.DataFrame,
    ticker: str = "",
    include_cross_asset: bool = True,
    include_alternative_data: bool = True,
    context: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute technical indicator features from OHLCV data.

    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume.
        ticker: Ticker symbol (used for sector-relative features).
        include_cross_asset: Whether to add market context features.
        include_alternative_data: Whether to add point-in-time public data.
        context: Pre-downloaded context data (avoids re-downloading).
    """
    feat = df.copy()

    # ── Core Returns ──
    for period in [1, 2, 3, 5, 10, 20]:
        feat[f"return_{period}d"] = feat["Close"].pct_change(period)

    # ── Moving Averages ──
    for window in [5, 10, 20, 50, 200]:
        feat[f"sma_{window}"] = feat["Close"].rolling(window).mean()
        feat[f"ema_{window}"] = feat["Close"].ewm(span=window, adjust=False).mean()
        feat[f"close_to_sma_{window}"] = feat["Close"] / feat[f"sma_{window}"] - 1

    # SMA crossover signals
    feat["sma_5_20_cross"] = (feat["sma_5"] > feat["sma_20"]).astype(int)
    feat["sma_20_50_cross"] = (feat["sma_20"] > feat["sma_50"]).astype(int)
    feat["sma_50_200_cross"] = (feat["sma_50"] > feat["sma_200"]).astype(int)

    # ── Volatility ──
    feat["volatility_5d"] = feat["return_1d"].rolling(5).std()
    feat["volatility_10d"] = feat["return_1d"].rolling(10).std()
    feat["volatility_20d"] = feat["return_1d"].rolling(20).std()
    feat["volatility_ratio"] = feat["volatility_5d"] / feat["volatility_20d"]

    # Parkinson volatility (uses high-low range, more accurate)
    feat["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2)))
        * ((np.log(feat["High"] / feat["Low"])) ** 2).rolling(20).mean()
    )

    # ── RSI (multiple periods) ──
    for period in [7, 14, 21]:
        delta = feat["Close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        feat[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # RSI divergence (price up but RSI down)
    feat["rsi_divergence"] = feat["return_5d"] - (feat["rsi_14"].diff(5) / 100)

    # ── MACD ──
    ema12 = feat["Close"].ewm(span=12, adjust=False).mean()
    ema26 = feat["Close"].ewm(span=26, adjust=False).mean()
    feat["macd"] = ema12 - ema26
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_histogram"] = feat["macd"] - feat["macd_signal"]
    feat["macd_cross"] = (feat["macd"] > feat["macd_signal"]).astype(int)

    # ── Bollinger Bands ──
    for window in [20, 50]:
        sma = feat["Close"].rolling(window).mean()
        std = feat["Close"].rolling(window).std()
        feat[f"bb_{window}_upper"] = sma + 2 * std
        feat[f"bb_{window}_lower"] = sma - 2 * std
        bb_range = feat[f"bb_{window}_upper"] - feat[f"bb_{window}_lower"]
        feat[f"bb_{window}_position"] = (feat["Close"] - feat[f"bb_{window}_lower"]) / bb_range
        feat[f"bb_{window}_width"] = bb_range / sma

    # ── Volume ──
    feat["volume_sma_10"] = feat["Volume"].rolling(10).mean()
    feat["volume_sma_20"] = feat["Volume"].rolling(20).mean()
    feat["volume_ratio"] = feat["Volume"] / feat["volume_sma_20"]
    feat["volume_trend"] = feat["volume_sma_10"] / feat["volume_sma_20"]

    # Volume-price confirmation
    feat["volume_price_trend"] = (
        feat["return_1d"] * feat["volume_ratio"]
    ).rolling(10).sum()

    # ── ATR ──
    high_low = feat["High"] - feat["Low"]
    high_close = (feat["High"] - feat["Close"].shift()).abs()
    low_close = (feat["Low"] - feat["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_ratio"] = feat["atr_14"] / feat["Close"]

    # ── Trend Strength / Directional Movement ──
    up_move = feat["High"].diff()
    down_move = -feat["Low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_sum = tr.rolling(14).sum()
    feat["plus_di_14"] = 100 * plus_dm.rolling(14).sum() / atr_sum
    feat["minus_di_14"] = 100 * minus_dm.rolling(14).sum() / atr_sum
    dx_denominator = feat["plus_di_14"] + feat["minus_di_14"]
    feat["adx_14"] = (
        100
        * (feat["plus_di_14"] - feat["minus_di_14"]).abs()
        / dx_denominator
    ).rolling(14).mean()

    # ── Commodity Channel / Money Flow ──
    typical_price = (feat["High"] + feat["Low"] + feat["Close"]) / 3
    typical_sma = typical_price.rolling(20).mean()
    mean_deviation = typical_price.rolling(20).apply(
        lambda values: np.mean(np.abs(values - values.mean())),
        raw=True,
    )
    feat["cci_20"] = (typical_price - typical_sma) / (0.015 * mean_deviation)
    raw_money_flow = typical_price * feat["Volume"]
    positive_flow = raw_money_flow.where(typical_price.diff() > 0, 0.0)
    negative_flow = raw_money_flow.where(typical_price.diff() < 0, 0.0)
    money_ratio = (
        positive_flow.rolling(14).sum()
        / negative_flow.rolling(14).sum()
    )
    feat["mfi_14"] = 100 - (100 / (1 + money_ratio))
    feat["chaikin_money_flow_20"] = (
        (
            ((feat["Close"] - feat["Low"]) - (feat["High"] - feat["Close"]))
            / (feat["High"] - feat["Low"])
        )
        * feat["Volume"]
    ).rolling(20).sum() / feat["Volume"].rolling(20).sum()

    # ── Stochastic ──
    for period in [14, 21]:
        low_n = feat["Low"].rolling(period).min()
        high_n = feat["High"].rolling(period).max()
        feat[f"stoch_k_{period}"] = ((feat["Close"] - low_n) / (high_n - low_n)) * 100
        feat[f"stoch_d_{period}"] = feat[f"stoch_k_{period}"].rolling(3).mean()
        feat[f"williams_r_{period}"] = (
            -100 * (high_n - feat["Close"]) / (high_n - low_n)
        )

    # ── Breakout / Channel Strategies ──
    donchian_high = feat["High"].rolling(20).max()
    donchian_low = feat["Low"].rolling(20).min()
    feat["donchian_position_20"] = (
        (feat["Close"] - donchian_low) / (donchian_high - donchian_low)
    )
    feat["donchian_breakout_up"] = (
        feat["Close"] >= donchian_high.shift(1)
    ).astype(int)
    feat["donchian_breakout_down"] = (
        feat["Close"] <= donchian_low.shift(1)
    ).astype(int)
    ema20 = feat["Close"].ewm(span=20, adjust=False).mean()
    feat["keltner_position"] = (
        (feat["Close"] - (ema20 - 2 * feat["atr_14"]))
        / (4 * feat["atr_14"])
    )

    # ── Ichimoku ──
    conversion = (
        feat["High"].rolling(9).max() + feat["Low"].rolling(9).min()
    ) / 2
    base = (
        feat["High"].rolling(26).max() + feat["Low"].rolling(26).min()
    ) / 2
    span_a = (conversion + base) / 2
    span_b = (
        feat["High"].rolling(52).max() + feat["Low"].rolling(52).min()
    ) / 2
    feat["ichimoku_conversion_base"] = conversion / base - 1
    feat["ichimoku_cloud_position"] = (
        (feat["Close"] - span_a) / (span_a - span_b).abs()
    )

    # ── OBV ──
    direction = np.sign(feat["Close"].diff())
    feat["obv"] = (direction * feat["Volume"]).cumsum()
    feat["obv_sma_10"] = feat["obv"].rolling(10).mean()
    feat["obv_slope"] = feat["obv"].diff(5) / feat["obv"].rolling(5).mean()

    # ── Range & Gap ──
    feat["range_pct"] = (feat["High"] - feat["Low"]) / feat["Close"]
    feat["gap_pct"] = feat["Open"] / feat["Close"].shift() - 1

    # ── Price Position ──
    feat["pct_from_52w_high"] = feat["Close"] / feat["High"].rolling(252).max() - 1
    feat["pct_from_52w_low"] = feat["Close"] / feat["Low"].rolling(252).min() - 1
    feat["pct_from_20d_high"] = feat["Close"] / feat["High"].rolling(20).max() - 1
    feat["pct_from_20d_low"] = feat["Close"] / feat["Low"].rolling(20).min() - 1
    feat["roc_10"] = feat["Close"].pct_change(10)
    feat["roc_20"] = feat["Close"].pct_change(20)
    feat["drawdown_60d"] = (
        feat["Close"] / feat["Close"].rolling(60).max() - 1
    )

    # ── Lag Features ──
    for lag in [1, 2, 3, 5, 10]:
        feat[f"return_lag_{lag}"] = feat["return_1d"].shift(lag)
    for lag in [1, 2, 3]:
        feat[f"volume_ratio_lag_{lag}"] = feat["volume_ratio"].shift(lag)

    # ── Rolling Statistics ──
    feat["sharpe_20d"] = (
        feat["return_1d"].rolling(20).mean() / feat["return_1d"].rolling(20).std()
    )
    feat["skew_20d"] = feat["return_1d"].rolling(20).skew()
    feat["kurt_20d"] = feat["return_1d"].rolling(20).kurt()
    feat["downside_volatility_20d"] = (
        feat["return_1d"].where(feat["return_1d"] < 0, 0).rolling(20).std()
    )
    feat["return_autocorr_20d"] = feat["return_1d"].rolling(20).corr(
        feat["return_1d"].shift(1)
    )
    feat["price_volume_corr_20d"] = feat["return_1d"].rolling(20).corr(
        feat["volume_ratio"].pct_change()
    )

    feat = feat.copy()

    # ── Streaks ──
    up_streak = (feat["return_1d"] > 0).astype(int)
    feat["up_streak"] = up_streak.groupby((up_streak != up_streak.shift()).cumsum()).cumsum()
    down_streak = (feat["return_1d"] < 0).astype(int)
    feat["down_streak"] = down_streak.groupby((down_streak != down_streak.shift()).cumsum()).cumsum()

    # ── Calendar ──
    if hasattr(feat.index, "dayofweek"):
        feat["day_of_week"] = feat.index.dayofweek
        feat["month"] = feat.index.month
    else:
        feat["day_of_week"] = 0
        feat["month"] = 0

    # ── Feature Interactions (top pairs from importance analysis) ──
    feat["rsi_x_volume"] = feat["rsi_14"] * feat["volume_ratio"]
    feat["momentum_x_vol"] = feat["return_20d"] * feat["volatility_20d"]
    feat["sma_spread_x_rsi"] = feat["close_to_sma_200"] * feat["rsi_14"]

    # ── Cross-Asset Context ──
    if include_cross_asset:
        if context is None:
            period = "5y"
            interval = "1d"
            context = _download_context(period, interval)
        feat = _add_cross_asset_features(feat, context, ticker)

    if include_alternative_data:
        feat = _add_alternative_data_features(feat, ticker)

    # ── Calendar Features ──
    if hasattr(feat.index, 'dayofweek'):
        feat["day_of_week"] = feat.index.dayofweek
        feat["month"] = feat.index.month
        feat["quarter"] = feat.index.quarter
        feat["is_month_start"] = feat.index.is_month_start.astype(int)
        feat["is_month_end"] = feat.index.is_month_end.astype(int)
        feat["is_quarter_end"] = feat.index.is_quarter_end.astype(int)
        feat["day_of_month"] = feat.index.day

    # ── Lag Features (use prior day's signals to predict next move) ──
    for lag in [1, 2, 3, 5]:
        feat[f"return_1d_lag{lag}"] = feat["return_1d"].shift(lag)
        feat[f"volume_ratio_lag{lag}"] = feat["volume_ratio"].shift(lag) if "volume_ratio" in feat.columns else np.nan

    # ── Cumulative return streaks ──
    if "return_1d" in feat.columns:
        up = (feat["return_1d"] > 0).astype(int)
        down = (feat["return_1d"] < 0).astype(int)
        feat["up_streak"] = up.groupby((up != up.shift()).cumsum()).cumcount() * up
        feat["down_streak"] = down.groupby((down != down.shift()).cumsum()).cumcount() * down

    # ── Drop raw columns ──
    drop_cols = {"Open", "High", "Low", "Close", "Volume", "Dividends",
                 "Stock Splits", "Capital Gains"}
    feature_cols = [c for c in feat.columns if c not in drop_cols]

    return feat[feature_cols].dropna()


def compute_intraday_features(
    df: pd.DataFrame,
    ticker: str = "",
    include_alternative_data: bool = True,
) -> pd.DataFrame:
    """Compute features for intraday (minute/hour) data.

    Simpler feature set that doesn't require 200+ periods of warmup.
    """
    feat = df.copy()

    # Short-term returns
    for period in [1, 3, 5, 10, 15, 30]:
        feat[f"return_{period}"] = feat["Close"].pct_change(period)

    # Short EMAs
    for window in [5, 10, 20, 50]:
        feat[f"ema_{window}"] = feat["Close"].ewm(span=window, adjust=False).mean()
        feat[f"close_to_ema_{window}"] = feat["Close"] / feat[f"ema_{window}"] - 1

    # Short-term RSI
    for period in [7, 14]:
        delta = feat["Close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        feat[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # Volatility
    feat["vol_5"] = feat["return_1"].rolling(5).std()
    feat["vol_20"] = feat["return_1"].rolling(20).std()
    feat["vol_ratio"] = feat["vol_5"] / feat["vol_20"]

    # Volume
    feat["volume_sma_20"] = feat["Volume"].rolling(20).mean()
    feat["volume_ratio"] = feat["Volume"] / feat["volume_sma_20"]

    # VWAP
    feat["cum_vol"] = feat["Volume"].cumsum()
    feat["cum_vp"] = (feat["Close"] * feat["Volume"]).cumsum()
    feat["vwap"] = feat["cum_vp"] / feat["cum_vol"]
    feat["close_to_vwap"] = feat["Close"] / feat["vwap"] - 1

    # Bollinger
    sma20 = feat["Close"].rolling(20).mean()
    std20 = feat["Close"].rolling(20).std()
    feat["bb_position"] = (feat["Close"] - (sma20 - 2 * std20)) / (4 * std20)

    # Spread/range
    feat["spread"] = (feat["High"] - feat["Low"]) / feat["Close"]

    # Lag features
    for lag in [1, 2, 3, 5]:
        feat[f"return_lag_{lag}"] = feat["return_1"].shift(lag)

    # Momentum
    feat["momentum_10"] = feat["Close"].pct_change(10)
    feat["momentum_30"] = feat["Close"].pct_change(30)

    # Stochastic
    low14 = feat["Low"].rolling(14).min()
    high14 = feat["High"].rolling(14).max()
    feat["stoch_k"] = ((feat["Close"] - low14) / (high14 - low14)) * 100

    if include_alternative_data:
        feat = _add_alternative_data_features(feat, ticker)

    drop_cols = {"Open", "High", "Low", "Close", "Volume", "Dividends",
                 "Stock Splits", "Capital Gains", "cum_vol", "cum_vp", "vwap"}
    feature_cols = [c for c in feat.columns if c not in drop_cols]
    return feat[feature_cols].dropna()


def compute_realtime_features(
    df: pd.DataFrame,
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the validated one-minute feature schema."""
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    feat = pd.DataFrame(index=df.index)

    for period in [1, 2, 3, 5, 10, 15, 30, 60]:
        feat[f"return_{period}"] = close.pct_change(period)

    returns = close.pct_change()
    for window in [5, 10, 20, 30, 60]:
        feat[f"mean_{window}"] = returns.rolling(window).mean()
        feat[f"vol_{window}"] = returns.rolling(window).std()
        feat[f"price_z_{window}"] = (
            close / close.rolling(window).mean() - 1
        )
        feat[f"volume_{window}"] = volume / volume.rolling(window).mean()

    delta = close.diff()
    for window in [7, 14, 28]:
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        feat[f"rsi_{window}"] = 100 - 100 / (1 + gain / loss)

    price_range = df["High"] - df["Low"]
    feat["range"] = price_range / close
    feat["close_location"] = (close - df["Low"]) / price_range
    feat["spread_open"] = close / df["Open"] - 1

    minute = df.index.hour * 60 + df.index.minute
    session_minute = minute - (9 * 60 + 30)
    feat["session_progress"] = session_minute / 390
    feat["session_sin"] = np.sin(2 * np.pi * session_minute / 390)
    feat["session_cos"] = np.cos(2 * np.pi * session_minute / 390)

    market_close = market_data["Close"].reindex(df.index).ffill()
    for period in [1, 3, 5, 10, 30]:
        market_return = market_close.pct_change(period)
        feat[f"market_return_{period}"] = market_return
        feat[f"relative_return_{period}"] = (
            close.pct_change(period) - market_return
        )
    feat["market_vol_20"] = market_close.pct_change().rolling(20).std()

    return feat.replace([np.inf, -np.inf], np.nan).dropna()


def create_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0) -> pd.Series:
    """Create forward-looking classification target.

    Args:
        df: DataFrame with 'Close' column.
        horizon: Number of periods to look ahead.
        threshold: Minimum return to classify as UP or DOWN.
    """
    future_return = df["Close"].pct_change(horizon).shift(-horizon)
    target = pd.Series(0, index=df.index, name="target")
    target[future_return > threshold] = 1
    target[future_return <= -threshold] = -1
    if threshold == 0:
        target[future_return == 0] = 1
    return target


def create_regression_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Create forward-looking regression target (% return)."""
    return df["Close"].pct_change(horizon).shift(-horizon).rename("target_return")


def create_intraday_targets(
    df: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> tuple[pd.Series, pd.Series]:
    """Create targets without allowing a horizon to cross a market session."""
    close = df["Close"]
    session = pd.Series(df.index.date, index=df.index)
    future_close = close.groupby(session).shift(-horizon)
    future_return = future_close / close - 1

    classification = pd.Series(0, index=df.index, name="target")
    classification[future_return > threshold] = 1
    classification[future_return <= -threshold] = -1
    if threshold == 0:
        classification[future_return == 0] = 1
    classification[future_return.isna()] = np.nan
    return classification, future_return.rename("target_return")
