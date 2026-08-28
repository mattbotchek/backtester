"""Stock Prediction ML Model.

Ensemble model combining:
  - Random Forest (classification + regression)
  - Gradient Boosting (XGBoost if available, else sklearn GBM)
  - A simple neural network (sklearn MLPClassifier)

The ensemble votes on direction (UP/DOWN/FLAT) and averages magnitude.
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
)

from .features import (
    _download_context,
    compute_features,
    create_regression_target,
    create_target,
)
from .compute import (
    accelerated_classifier,
    accelerated_regressor,
    compute_backend,
    model_worker_count,
    parallel_map,
)
from .prediction_store import (
    PredictionStore,
    iso_utc,
    prediction_target_time,
    utc_now,
)
from .persistence import atomic_pickle_dump

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def reconcile_prediction(
    class_label: int,
    predicted_return: float,
) -> tuple[str, float]:
    """Make the deployed direction and price target agree."""
    if class_label > 0:
        return "UP", abs(float(predicted_return))
    if class_label < 0:
        return "DOWN", -abs(float(predicted_return))
    return "FLAT", 0.0


@dataclass
class Prediction:
    """A single stock prediction."""

    ticker: str
    direction: str  # "UP", "DOWN", "FLAT"
    predicted_return_pct: float
    confidence: float  # 0-1
    outliers: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: iso_utc(utc_now()))
    predicted_for: str = ""
    input_at: str = ""
    base_price: float = 0.0
    predicted_price: float = 0.0
    model_name: str = "stock_predictor"
    last_trained_at: str | None = None
    forward_accuracy: float | None = None
    settled_predictions: int = 0

    def __repr__(self):
        arrow = {"UP": "^", "DOWN": "v", "FLAT": "-"}.get(self.direction, "?")
        return (
            f"{self.ticker} {arrow} {self.predicted_return_pct:+.2f}% "
            f"to ${self.predicted_price:.2f} by {self.predicted_for} "
            f"(confidence: {self.confidence:.0%}) "
            f"{'Outliers: ' + ', '.join(self.outliers) if self.outliers else ''}"
        )


class StockPredictor:
    """Ensemble ML model for stock price prediction.

    Usage:
        predictor = StockPredictor()
        predictor.train(tickers=["AAPL", "MSFT", ...], timeline="5y")
        prediction = predictor.predict("TSLA")
    """

    def __init__(
        self,
        horizon: int = 10,
        threshold: float = 0.0,
        model_name: str = "stock_predictor",
    ):
        self.horizon = horizon
        self.threshold = threshold
        self.model_name = model_name
        self.interval = "1d"
        self.scaler = StandardScaler()

        # Binary classification ensemble (UP=1 vs DOWN=0)
        self.classifier = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=300, max_depth=8, min_samples_leaf=30,
                    max_features="sqrt",
                    random_state=42, n_jobs=model_worker_count()
                )),
                ("hgb", accelerated_classifier(
                    HistGradientBoostingClassifier(
                        max_iter=300, max_depth=6, min_samples_leaf=30,
                        learning_rate=0.03, max_bins=128,
                        l2_regularization=1.0,
                        early_stopping=True, validation_fraction=0.1,
                        random_state=42
                    ),
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.03,
                    min_child_weight=30,
                    reg_lambda=1.0,
                    random_state=42,
                )),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32), max_iter=500,
                    early_stopping=True, random_state=42
                )),
            ],
            voting="soft",
            n_jobs=model_worker_count(),
        )

        # Regression model for magnitude
        self.regressor = accelerated_regressor(
            HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=5,
                min_samples_leaf=20,
                learning_rate=0.03,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ),
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            min_child_weight=20,
            reg_lambda=1.0,
            random_state=42,
        )

        self.feature_names: list[str] = []
        self.is_trained = False
        self._training_stats: dict = {}

    def _prepare_data(self, tickers: list[str], timeline: str = "5y",
                      interval: str = "1d") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Download data and compute features + targets for training."""
        import yfinance

        all_features = []
        all_cls_targets = []
        all_reg_targets = []
        all_dates = []
        context = _download_context(timeline, interval)

        def prepare_ticker(ticker: str):
            try:
                logger.info("Downloading %s...", ticker)
                data = yfinance.Ticker(ticker).history(period=timeline, interval=interval)
                if data.empty or len(data) < 250:
                    logger.warning("Insufficient data for %s, skipping.", ticker)
                    return None

                features = compute_features(
                    data,
                    ticker=ticker,
                    include_cross_asset=True,
                    context=context,
                )
                cls_target = create_target(data, horizon=self.horizon, threshold=self.threshold)
                reg_target = create_regression_target(data, horizon=self.horizon)

                # Align indices
                common_idx = features.index.intersection(cls_target.dropna().index).intersection(
                    reg_target.dropna().index
                )
                if len(common_idx) < 100:
                    return None

                return (
                    features.loc[common_idx],
                    cls_target.loc[common_idx],
                    reg_target.loc[common_idx],
                )
            except Exception as e:
                logger.warning("Error processing %s: %s", ticker, e)
                return None

        for prepared in parallel_map(prepare_ticker, tickers):
            if prepared is None:
                continue
            features, cls_target, reg_target = prepared
            all_features.append(features)
            all_cls_targets.append(cls_target)
            all_reg_targets.append(reg_target)
            all_dates.append(pd.Series(features.index))

        if not all_features:
            raise ValueError("No valid training data collected.")

        # Reset indices to avoid duplicate datetime indices across tickers
        X = pd.concat(all_features, ignore_index=True)
        y_cls = pd.concat(all_cls_targets, ignore_index=True)
        y_reg = pd.concat(all_reg_targets, ignore_index=True)
        dates = pd.concat(all_dates, ignore_index=True)

        # Drop any remaining NaN/inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = X.notna().all(axis=1) & y_cls.notna() & y_reg.notna()
        X = X[mask].reset_index(drop=True)
        y_cls = y_cls[mask].reset_index(drop=True)
        y_reg = y_reg[mask].reset_index(drop=True)
        dates = dates[mask].reset_index(drop=True)

        order = dates.sort_values(kind="stable").index
        X = X.loc[order].reset_index(drop=True)
        y_cls = y_cls.loc[order].reset_index(drop=True)
        y_reg = y_reg.loc[order].reset_index(drop=True)

        return X, y_cls, y_reg

    def train(self, tickers: list[str], timeline: str = "5y",
              interval: str = "1d") -> dict:
        """Train the model on historical data from multiple tickers.

        Returns training statistics including cross-validation scores.
        """
        logger.info("Preparing training data for %d tickers...", len(tickers))
        X, y_cls, y_reg = self._prepare_data(tickers, timeline, interval)

        self.feature_names = list(X.columns)
        logger.info("Training with %d samples, %d features", len(X), len(self.feature_names))

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_cls.iloc[train_idx], y_cls.iloc[val_idx]

            self.classifier.fit(X_train, y_train)
            score = accuracy_score(y_val, self.classifier.predict(X_val))
            cv_scores.append(score)

        # Final fit on all data
        self.classifier.fit(X_scaled, y_cls)
        self.regressor.fit(X_scaled, y_reg)
        self.is_trained = True

        self._training_stats = {
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "n_tickers": len(tickers),
            "cv_accuracy_mean": np.mean(cv_scores),
            "cv_accuracy_std": np.std(cv_scores),
            "cv_scores": cv_scores,
            "class_distribution": dict(y_cls.value_counts()),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "compute_backend": compute_backend(),
        }

        logger.info(
            "Training complete. CV accuracy: %.3f ± %.3f",
            self._training_stats["cv_accuracy_mean"],
            self._training_stats["cv_accuracy_std"],
        )

        return self._training_stats

    def predict(self, ticker: str, timeline: str = "2y",
                interval: str = "1d") -> Prediction:
        """Generate a prediction for a single ticker.

        Uses the most recent data point's features.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")

        import yfinance
        data = yfinance.Ticker(ticker).history(period=timeline, interval=interval)
        if data.empty:
            raise ValueError(f"No data available for {ticker}")

        features = compute_features(data, ticker=ticker)
        if features.empty:
            raise ValueError(f"Insufficient data to compute features for {ticker}")

        # Use last row (most recent)
        X = features.iloc[[-1]][self.feature_names]
        X_scaled = self.scaler.transform(X)

        # Classification supplies direction; regression supplies magnitude.
        class_label = int(self.classifier.predict(X_scaled)[0])
        raw_return = float(self.regressor.predict(X_scaled)[0])
        proba = self.classifier.predict_proba(X_scaled)[0]
        confidence = float(max(proba))
        direction, predicted_return_fraction = reconcile_prediction(
            class_label,
            raw_return,
        )
        predicted_return = predicted_return_fraction * 100

        # Outlier detection
        outliers = self._detect_outliers(features.iloc[-1])

        input_at = features.index[-1]
        target_at = prediction_target_time(input_at, self.horizon, interval)
        base_price = float(data.loc[input_at, "Close"])
        predicted_price = base_price * (1 + predicted_return / 100)
        store = PredictionStore()
        stored = store.record_prediction(
            model_name=self.model_name,
            family="StockPredictor",
            ticker=ticker,
            input_at=input_at,
            target_at=target_at,
            interval=interval,
            horizon=self.horizon,
            base_price=base_price,
            predicted_return_pct=predicted_return,
            direction=direction,
            confidence=confidence,
            last_trained_at=self._training_stats.get("trained_at"),
            training_metrics=self._training_stats,
            metadata={"outliers": outliers},
        )
        stats = store.model_statistics(self.model_name)
        model_stats = stats[0] if stats else {}

        return Prediction(
            ticker=ticker,
            direction=direction,
            predicted_return_pct=predicted_return,
            confidence=confidence,
            outliers=outliers,
            timestamp=stored["created_at"],
            predicted_for=stored["target_at"],
            input_at=stored["input_at"],
            base_price=stored["base_price"],
            predicted_price=stored["predicted_price"],
            model_name=self.model_name,
            last_trained_at=model_stats.get("last_trained_at"),
            forward_accuracy=model_stats.get("forward_accuracy"),
            settled_predictions=model_stats.get("settled_count", 0),
        )

    def _detect_outliers(self, row: pd.Series) -> list[str]:
        """Identify features that are at extreme values."""
        outliers = []
        if "rsi_14" in row and not np.isnan(row["rsi_14"]):
            if row["rsi_14"] > 80:
                outliers.append(f"RSI extremely overbought ({row['rsi_14']:.0f})")
            elif row["rsi_14"] < 20:
                outliers.append(f"RSI extremely oversold ({row['rsi_14']:.0f})")

        if "volume_ratio" in row and not np.isnan(row["volume_ratio"]):
            if row["volume_ratio"] > 3:
                outliers.append(f"Volume {row['volume_ratio']:.1f}x above average")

        if "bb_position" in row and not np.isnan(row["bb_position"]):
            if row["bb_position"] > 1:
                outliers.append("Price above upper Bollinger Band")
            elif row["bb_position"] < 0:
                outliers.append("Price below lower Bollinger Band")

        if "volatility_20d" in row and not np.isnan(row["volatility_20d"]):
            if row["volatility_20d"] > 0.04:
                outliers.append(f"High volatility ({row['volatility_20d']:.1%} daily)")

        if "gap_pct" in row and not np.isnan(row["gap_pct"]):
            if abs(row["gap_pct"]) > 0.03:
                outliers.append(f"Large gap {'up' if row['gap_pct'] > 0 else 'down'} ({row['gap_pct']:.1%})")

        return outliers

    def save(self, name: str = "stock_predictor"):
        """Save the trained model to disk."""
        self.model_name = name
        path = MODEL_DIR / f"{name}.pkl"
        atomic_pickle_dump({
            "classifier": self.classifier,
            "regressor": self.regressor,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_stats": self._training_stats,
            "horizon": self.horizon,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }, path)
        PredictionStore().register_model(
            model_name=self.model_name,
            family="StockPredictor",
            horizon=self.horizon,
            interval=self.interval,
            last_trained_at=self._training_stats.get("trained_at"),
            training_metrics=self._training_stats,
        )
        logger.info("Model saved to %s", path)

    def load(self, name: str = "stock_predictor"):
        """Load a trained model from disk."""
        path = MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.regressor = data["regressor"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self._training_stats = data["training_stats"]
        self.horizon = data["horizon"]
        self.threshold = data["threshold"]
        self.model_name = data.get("model_name", name)
        self.is_trained = True
        PredictionStore().register_model(
            model_name=self.model_name,
            family="StockPredictor",
            horizon=self.horizon,
            interval=self.interval,
            last_trained_at=self._training_stats.get("trained_at"),
            training_metrics=self._training_stats,
        )
        logger.info("Model loaded from %s", path)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances from the Random Forest sub-model."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        rf = self.classifier.named_estimators_["rf"]
        importances = rf.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
