"""Feature engineering — computes technical indicators for ML input.

Transforms raw OHLCV data into a feature matrix that the ML model
consumes. All features are look-back only (no future data leakage).
"""

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features from OHLCV data.

    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume (indexed by date).

    Returns:
        DataFrame with feature columns appended (NaN rows from lookback dropped).
    """
    feat = df.copy()

    # Returns
    feat["return_1d"] = feat["Close"].pct_change(1)
    feat["return_5d"] = feat["Close"].pct_change(5)
    feat["return_10d"] = feat["Close"].pct_change(10)
    feat["return_20d"] = feat["Close"].pct_change(20)

    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        feat[f"sma_{window}"] = feat["Close"].rolling(window).mean()
        feat[f"ema_{window}"] = feat["Close"].ewm(span=window, adjust=False).mean()
        feat[f"close_to_sma_{window}"] = feat["Close"] / feat[f"sma_{window}"] - 1

    # Volatility
    feat["volatility_10d"] = feat["return_1d"].rolling(10).std()
    feat["volatility_20d"] = feat["return_1d"].rolling(20).std()

    # RSI
    delta = feat["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = feat["Close"].ewm(span=12, adjust=False).mean()
    ema26 = feat["Close"].ewm(span=26, adjust=False).mean()
    feat["macd"] = ema12 - ema26
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_histogram"] = feat["macd"] - feat["macd_signal"]

    # Bollinger Bands
    sma20 = feat["Close"].rolling(20).mean()
    std20 = feat["Close"].rolling(20).std()
    feat["bb_upper"] = sma20 + 2 * std20
    feat["bb_lower"] = sma20 - 2 * std20
    feat["bb_position"] = (feat["Close"] - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"])

    # Volume features
    feat["volume_sma_20"] = feat["Volume"].rolling(20).mean()
    feat["volume_ratio"] = feat["Volume"] / feat["volume_sma_20"]

    # ATR
    high_low = feat["High"] - feat["Low"]
    high_close = (feat["High"] - feat["Close"].shift()).abs()
    low_close = (feat["Low"] - feat["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_ratio"] = feat["atr_14"] / feat["Close"]

    # Stochastic
    low14 = feat["Low"].rolling(14).min()
    high14 = feat["High"].rolling(14).max()
    feat["stoch_k"] = ((feat["Close"] - low14) / (high14 - low14)) * 100
    feat["stoch_d"] = feat["stoch_k"].rolling(3).mean()

    # OBV
    obv = [0]
    for i in range(1, len(feat)):
        if feat["Close"].iloc[i] > feat["Close"].iloc[i - 1]:
            obv.append(obv[-1] + feat["Volume"].iloc[i])
        elif feat["Close"].iloc[i] < feat["Close"].iloc[i - 1]:
            obv.append(obv[-1] - feat["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    feat["obv"] = obv
    feat["obv_sma_10"] = pd.Series(obv, index=feat.index).rolling(10).mean()

    # Range features
    feat["range_pct"] = (feat["High"] - feat["Low"]) / feat["Close"]
    feat["gap_pct"] = feat["Open"] / feat["Close"].shift() - 1

    # Day of week (0=Monday)
    if hasattr(feat.index, "dayofweek"):
        feat["day_of_week"] = feat.index.dayofweek
    else:
        feat["day_of_week"] = 0

    # Lag features (crucial for short-term prediction)
    for lag in [1, 2, 3, 5]:
        feat[f"return_lag_{lag}"] = feat["return_1d"].shift(lag)
        feat[f"volume_ratio_lag_{lag}"] = feat["volume_ratio"].shift(lag)

    # Rolling Sharpe (risk-adjusted return signal)
    feat["sharpe_20d"] = (
        feat["return_1d"].rolling(20).mean() / feat["return_1d"].rolling(20).std()
    )

    # Price position relative to 52-week high/low
    feat["pct_from_52w_high"] = feat["Close"] / feat["High"].rolling(252).max() - 1
    feat["pct_from_52w_low"] = feat["Close"] / feat["Low"].rolling(252).min() - 1

    # Consecutive up/down days
    up_streak = (feat["return_1d"] > 0).astype(int)
    feat["up_streak"] = up_streak.groupby((up_streak != up_streak.shift()).cumsum()).cumsum()
    down_streak = (feat["return_1d"] < 0).astype(int)
    feat["down_streak"] = down_streak.groupby((down_streak != down_streak.shift()).cumsum()).cumsum()

    # Drop raw OHLCV, intermediate, and yfinance metadata columns
    drop_cols = {"Open", "High", "Low", "Close", "Volume", "Dividends",
                 "Stock Splits", "Capital Gains"}
    feature_cols = [c for c in feat.columns if c not in drop_cols]

    return feat[feature_cols].dropna()


def create_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0) -> pd.Series:
    """Create forward-looking classification target.

    Args:
        df: DataFrame with 'Close' column.
        horizon: Number of days to look ahead.
        threshold: Minimum return to classify as UP or DOWN.
                   Default 0.0 = binary (UP=1 if positive, DOWN=-1 if negative).

    Returns:
        Series with values: 1 (UP), -1 (DOWN). FLAT (0) only if threshold > 0.
    """
    future_return = df["Close"].pct_change(horizon).shift(-horizon)
    target = pd.Series(0, index=df.index, name="target")
    target[future_return > threshold] = 1
    target[future_return <= -threshold] = -1
    # For binary (threshold=0), eliminate exact 0 returns
    if threshold == 0:
        target[future_return == 0] = 1  # tie-break to UP
    return target


def create_regression_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Create forward-looking regression target (% return)."""
    return df["Close"].pct_change(horizon).shift(-horizon).rename("target_return")
