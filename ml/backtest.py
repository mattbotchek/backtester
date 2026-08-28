"""ML Backtester — walk-forward testing without future data leakage.

Simulates real-world usage: at each point in time, the model only knows
past data. Trains on a rolling window across ALL tickers simultaneously,
then predicts the next period.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .features import compute_features, create_target, create_regression_target
from .model import StockPredictor, Prediction, reconcile_prediction
from .prediction_store import directional_hit

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Results from a walk-forward backtest."""

    predictions: list[dict] = field(default_factory=list)
    accuracy: float = 0.0
    directional_accuracy: float = 0.0
    mean_absolute_error: float = 0.0
    target_hit_accuracy: float = 0.0
    confidence_floor: float = 0.0
    coverage: float = 0.0
    eligible_predictions: int = 0
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
            f"  Target-price hit rate:   {self.target_hit_accuracy:.2%}\n"
            f"  Confidence floor:        {self.confidence_floor:.0%}\n"
            f"  Eligible predictions:    {self.eligible_predictions} "
            f"({self.coverage:.2%} coverage)\n"
            f"  Mean absolute error:     {self.mean_absolute_error:.4f}\n"
            f"\n{self.classification_report}\n"
            f"{'=' * 60}\n"
        )


def walk_forward_backtest(
    tickers: list[str],
    test_tickers: list[str] | None = None,
    train_period: str = "5y",
    total_history: str = "5y",
    horizon: int = 10,
    threshold: float = 0.0,
    n_splits: int = 5,
    min_train_size: int = 500,
    confidence_floor: float = 0.0,
) -> WalkForwardResult:
    """Run a walk-forward (expanding window) backtest across ALL tickers.

    Trains on all tickers' data up to time T, then tests on all tickers
    from T to T+window. This mirrors real deployment where the model is
    trained on broad market data.
    """
    if test_tickers is None:
        test_tickers = tickers

    if not 0 <= confidence_floor <= 1:
        raise ValueError("confidence_floor must be between 0 and 1")
    result = WalkForwardResult(confidence_floor=confidence_floor)

    # Step 1: Download and prepare all data upfront
    logger.info("Downloading data for %d tickers...", len(tickers))
    ticker_data: dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]] = {}

    for ticker in set(tickers) | set(test_tickers):
        try:
            data = yfinance.Ticker(ticker).history(period=total_history, interval="1d")
            if data.empty or len(data) < 300:
                logger.warning("Insufficient data for %s, skipping.", ticker)
                continue

            features = compute_features(data, ticker=ticker)
            cls_target = create_target(data, horizon=horizon, threshold=threshold)
            reg_target = create_regression_target(data, horizon=horizon)

            common_idx = (
                features.index
                .intersection(cls_target.dropna().index)
                .intersection(reg_target.dropna().index)
            )
            if len(common_idx) < 200:
                continue

            features = features.loc[common_idx]
            cls_target = cls_target.loc[common_idx]
            reg_target = reg_target.loc[common_idx]

            # Replace inf with NaN and drop
            features = features.replace([np.inf, -np.inf], np.nan).dropna()
            common_idx = features.index.intersection(cls_target.index).intersection(reg_target.index)
            features = features.loc[common_idx]
            cls_target = cls_target.loc[common_idx]
            reg_target = reg_target.loc[common_idx]

            ticker_data[ticker] = (features, cls_target, reg_target)
            logger.info("  %s: %d samples", ticker, len(features))
        except Exception as e:
            logger.warning("Error downloading %s: %s", ticker, e)

    if not ticker_data:
        logger.error("No valid data collected.")
        return result

    # Step 2: Find common date range across all tickers
    all_dates = sorted(set().union(*(d[0].index for d in ticker_data.values())))
    total_dates = len(all_dates)
    logger.info("Total unique dates: %d", total_dates)

    # Step 3: Walk-forward splits based on dates
    split_size = (total_dates - min_train_size) // n_splits
    if split_size < 10:
        logger.error("Not enough data for %d splits.", n_splits)
        return result

    all_true = []
    all_pred = []
    all_true_returns = []
    all_pred_returns = []
    all_target_hits = []
    total_candidates = 0

    for i in range(n_splits):
        train_end_idx = min_train_size + i * split_size
        test_end_idx = min(train_end_idx + split_size, total_dates)
        if test_end_idx <= train_end_idx:
            continue

        split_date = all_dates[train_end_idx]
        test_end_date = all_dates[test_end_idx - 1]
        purge_date = all_dates[max(0, train_end_idx - horizon)]
        logger.info(
            "Split %d/%d: train up to %s, test to %s",
            i + 1, n_splits, split_date, test_end_date
        )

        # Collect training data from ALL tickers
        train_X_parts, train_y_cls_parts, train_y_reg_parts = [], [], []
        for ticker in tickers:
            if ticker not in ticker_data:
                continue
            feat, cls_t, reg_t = ticker_data[ticker]
            # Exclude samples whose forward targets can overlap the test fold.
            train_mask = feat.index < purge_date
            if train_mask.sum() > 50:
                train_X_parts.append(feat[train_mask])
                train_y_cls_parts.append(cls_t[train_mask])
                train_y_reg_parts.append(reg_t[train_mask])

        if not train_X_parts:
            continue

        X_train = pd.concat(train_X_parts, ignore_index=True)
        y_cls_train = pd.concat(train_y_cls_parts, ignore_index=True)
        y_reg_train = pd.concat(train_y_reg_parts, ignore_index=True)

        # Train model on ALL tickers
        predictor = StockPredictor(horizon=horizon, threshold=threshold)
        predictor.feature_names = list(X_train.columns)
        predictor.scaler = StandardScaler()
        X_train_scaled = predictor.scaler.fit_transform(X_train)
        predictor.classifier.fit(X_train_scaled, y_cls_train)
        predictor.regressor.fit(X_train_scaled, y_reg_train)

        # Test on test tickers
        for ticker in test_tickers:
            if ticker not in ticker_data:
                continue
            feat, cls_t, reg_t = ticker_data[ticker]
            test_mask = (feat.index >= split_date) & (feat.index <= test_end_date)
            if test_mask.sum() == 0:
                continue

            X_test = feat[test_mask]
            y_cls_test = cls_t[test_mask]
            y_reg_test = reg_t[test_mask]

            # Align feature columns
            missing_cols = set(predictor.feature_names) - set(X_test.columns)
            if missing_cols:
                continue
            X_test = X_test[predictor.feature_names]

            X_test_scaled = predictor.scaler.transform(X_test)
            cls_preds = predictor.classifier.predict(X_test_scaled)
            raw_reg_preds = predictor.regressor.predict(X_test_scaled)
            probabilities = predictor.classifier.predict_proba(X_test_scaled)
            confidence = probabilities.max(axis=1)
            selected = confidence >= confidence_floor
            total_candidates += len(X_test)
            if not selected.any():
                continue

            positions = np.flatnonzero(selected)
            cls_preds = cls_preds[selected]
            raw_reg_preds = raw_reg_preds[selected]
            X_test = X_test.iloc[positions]
            y_cls_test = y_cls_test.iloc[positions]
            y_reg_test = y_reg_test.iloc[positions]
            confidence = confidence[selected]
            reconciled = [
                reconcile_prediction(int(label), float(predicted_return))
                for label, predicted_return in zip(cls_preds, raw_reg_preds)
            ]
            directions = [item[0] for item in reconciled]
            reg_preds = np.array([item[1] for item in reconciled])
            target_hits = [
                directional_hit(
                    direction,
                    1 + float(predicted_return),
                    1 + float(actual_return),
                )
                for direction, predicted_return, actual_return in zip(
                    directions,
                    reg_preds,
                    y_reg_test,
                )
            ]

            all_true.extend(y_cls_test.values)
            all_pred.extend(cls_preds)
            all_true_returns.extend(y_reg_test.values)
            all_pred_returns.extend(reg_preds)
            all_target_hits.extend(target_hits)

            for j in range(len(X_test)):
                result.predictions.append({
                    "ticker": ticker,
                    "date": str(X_test.index[j]),
                    "true_direction": int(y_cls_test.iloc[j]),
                    "pred_direction": int(cls_preds[j]),
                    "true_return": float(y_reg_test.iloc[j]),
                    "pred_return": float(reg_preds[j]),
                    "direction": directions[j],
                    "confidence": float(confidence[j]),
                    "target_hit": bool(target_hits[j]),
                })

    # Step 4: Compute aggregate metrics
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
        result.target_hit_accuracy = float(np.mean(all_target_hits))
        result.eligible_predictions = len(all_true)
        result.coverage = (
            len(all_true) / total_candidates if total_candidates else 0.0
        )

        label_map = {-1: "DOWN", 0: "FLAT", 1: "UP"}
        labels = sorted(set(all_true) | set(all_pred))
        target_names = [label_map.get(l, str(l)) for l in labels]
        result.classification_report = classification_report(
            all_true, all_pred, labels=labels, target_names=target_names,
            zero_division=0,
        )
        result.confusion_matrix = confusion_matrix(all_true, all_pred, labels=labels).tolist()

    return result
