"""ML Backtester — walk-forward testing without future data leakage.

Simulates real-world usage: at each point in time, the model only knows
past data. Trains on a rolling window and predicts the next period.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .features import compute_features, create_target, create_regression_target
from .model import StockPredictor, Prediction

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from a walk-forward backtest."""

    predictions: list[dict] = field(default_factory=list)
    accuracy: float = 0.0
    directional_accuracy: float = 0.0
    mean_absolute_error: float = 0.0
    classification_report: str = ""
    confusion_matrix: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"  ML WALK-FORWARD BACKTEST RESULTS\n"
            f"{'=' * 60}\n"
            f"  Total predictions:      {len(self.predictions)}\n"
            f"  Classification accuracy: {self.accuracy:.2%}\n"
            f"  Directional accuracy:    {self.directional_accuracy:.2%}\n"
            f"  Mean absolute error:     {self.mean_absolute_error:.4f}\n"
            f"\n{self.classification_report}\n"
            f"{'=' * 60}\n"
        )


def walk_forward_backtest(
    tickers: list[str],
    test_tickers: list[str] | None = None,
    train_period: str = "3y",
    total_history: str = "5y",
    horizon: int = 5,
    threshold: float = 0.02,
    n_splits: int = 5,
    min_train_size: int = 500,
) -> WalkForwardResult:
    """Run a walk-forward (expanding window) backtest.

    Splits data chronologically: trains on [0:t], predicts [t:t+horizon],
    then expands training window forward.

    Args:
        tickers: Tickers used for training.
        test_tickers: Tickers to test on (defaults to training tickers).
        train_period: Not used directly — the split is done on downloaded data.
        total_history: How much history to download.
        horizon: Prediction horizon in days.
        threshold: Classification threshold for UP/DOWN.
        n_splits: Number of walk-forward splits.
        min_train_size: Minimum training samples before first prediction.
    """
    if test_tickers is None:
        test_tickers = tickers

    result = WalkForwardResult()
    all_true = []
    all_pred = []
    all_true_returns = []
    all_pred_returns = []

    for ticker in test_tickers:
        try:
            logger.info("Walk-forward testing %s...", ticker)
            data = yfinance.Ticker(ticker).history(period=total_history, interval="1d")
            if data.empty or len(data) < min_train_size + horizon * n_splits:
                logger.warning("Insufficient data for %s.", ticker)
                continue

            features = compute_features(data)
            cls_target = create_target(data, horizon=horizon, threshold=threshold)
            reg_target = create_regression_target(data, horizon=horizon)

            common_idx = (
                features.index
                .intersection(cls_target.dropna().index)
                .intersection(reg_target.dropna().index)
            )
            features = features.loc[common_idx]
            cls_target = cls_target.loc[common_idx]
            reg_target = reg_target.loc[common_idx]

            total = len(features)
            split_size = (total - min_train_size) // n_splits

            for i in range(n_splits):
                train_end = min_train_size + i * split_size
                test_start = train_end
                test_end = min(test_start + split_size, total)

                if test_end <= test_start:
                    continue

                X_train = features.iloc[:train_end]
                y_cls_train = cls_target.iloc[:train_end]
                y_reg_train = reg_target.iloc[:train_end]
                X_test = features.iloc[test_start:test_end]
                y_cls_test = cls_target.iloc[test_start:test_end]
                y_reg_test = reg_target.iloc[test_start:test_end]

                # Train a fresh model on the expanding window
                predictor = StockPredictor(horizon=horizon, threshold=threshold)
                predictor.feature_names = list(X_train.columns)

                from sklearn.preprocessing import StandardScaler
                predictor.scaler = StandardScaler()
                X_train_scaled = predictor.scaler.fit_transform(X_train)
                X_test_scaled = predictor.scaler.transform(X_test)

                predictor.classifier.fit(X_train_scaled, y_cls_train)
                predictor.regressor.fit(X_train_scaled, y_reg_train)

                cls_preds = predictor.classifier.predict(X_test_scaled)
                reg_preds = predictor.regressor.predict(X_test_scaled)

                all_true.extend(y_cls_test.values)
                all_pred.extend(cls_preds)
                all_true_returns.extend(y_reg_test.values)
                all_pred_returns.extend(reg_preds)

                for j in range(len(X_test)):
                    result.predictions.append({
                        "ticker": ticker,
                        "date": str(X_test.index[j]),
                        "true_direction": int(y_cls_test.iloc[j]),
                        "pred_direction": int(cls_preds[j]),
                        "true_return": float(y_reg_test.iloc[j]),
                        "pred_return": float(reg_preds[j]),
                    })

        except Exception as e:
            logger.warning("Error in walk-forward for %s: %s", ticker, e)

    if all_true:
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        result.accuracy = accuracy_score(all_true, all_pred)

        # Directional accuracy (ignoring FLAT)
        mask = (all_true != 0) | (all_pred != 0)
        if mask.sum() > 0:
            dir_true = np.sign(all_true[mask])
            dir_pred = np.sign(all_pred[mask])
            result.directional_accuracy = accuracy_score(dir_true, dir_pred)

        result.mean_absolute_error = float(np.mean(np.abs(
            np.array(all_true_returns) - np.array(all_pred_returns)
        )))

        label_map = {-1: "DOWN", 0: "FLAT", 1: "UP"}
        labels = sorted(set(all_true) | set(all_pred))
        target_names = [label_map.get(l, str(l)) for l in labels]
        result.classification_report = classification_report(
            all_true, all_pred, labels=labels, target_names=target_names,
            zero_division=0,
        )
        result.confusion_matrix = confusion_matrix(all_true, all_pred, labels=labels).tolist()

    return result
