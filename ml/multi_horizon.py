"""Multi-Horizon Stock Prediction Models.

Creates and manages separate models optimized for different timeframes:
  - Intraday (day trading): minute/hour intervals, 1-60 min horizon
  - Short term: 1-5 day horizon
  - Medium term: 10-30 day horizon
  - Long term: 60-252 day (1 year) horizon

Each model uses different features, hyperparameters, and training
approaches tuned for its specific timeframe.
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from .features import (
    compute_features,
    compute_intraday_features,
    compute_realtime_features,
    create_target,
    create_regression_target,
    create_intraday_targets,
    _download_context,
)
from .model import reconcile_prediction
from .compute import (
    accelerated_classifier,
    accelerated_regressor,
    compute_backend,
    model_worker_count,
    parallel_map,
)
from .prediction_store import (
    PredictionStore,
    directional_hit,
    prediction_target_time,
)
from .persistence import atomic_pickle_dump

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


@dataclass
class HorizonConfig:
    """Configuration for a specific prediction horizon."""

    name: str
    horizon: int  # periods ahead
    threshold: float  # classification threshold
    data_period: str  # yfinance period for download
    data_interval: str  # yfinance interval
    min_train_samples: int
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 30
    learning_rate: float = 0.03
    l2_reg: float = 1.0
    use_cross_asset: bool = True
    use_intraday_features: bool = False
    use_realtime_features: bool = False
    classifier_kind: str = "ensemble"
    target_price_mode: str = "regression"

    @property
    def model_file(self) -> str:
        return f"model_{self.name}"


# Pre-configured horizons
HORIZONS = {
    "realtime_5m": HorizonConfig(
        name="realtime_5m",
        horizon=5,
        threshold=0.0025,
        data_period="7d",
        data_interval="1m",
        min_train_samples=500,
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        use_cross_asset=False,
        use_realtime_features=True,
        classifier_kind="extra_trees",
        target_price_mode="threshold",
    ),
    "intraday": HorizonConfig(
        name="intraday",
        horizon=30,  # 30 bars (minutes) ahead
        threshold=0.001,  # 0.1% move
        data_period="60d",
        data_interval="5m",
        min_train_samples=200,
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        learning_rate=0.05,
        l2_reg=2.0,
        use_cross_asset=False,
        use_intraday_features=True,
    ),
    "short_1d": HorizonConfig(
        name="short_1d",
        horizon=1,
        threshold=0.0,
        data_period="5y",
        data_interval="1d",
        min_train_samples=500,
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=40,
        learning_rate=0.02,
        l2_reg=2.0,
    ),
    "short_5d": HorizonConfig(
        name="short_5d",
        horizon=5,
        threshold=0.005,
        data_period="5y",
        data_interval="1d",
        min_train_samples=500,
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=35,
        learning_rate=0.02,
        l2_reg=1.5,
    ),
    "medium_20d": HorizonConfig(
        name="medium_20d",
        horizon=20,
        threshold=0.01,
        data_period="10y",
        data_interval="1d",
        min_train_samples=500,
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=30,
        learning_rate=0.02,
        l2_reg=1.0,
    ),
    "long_60d": HorizonConfig(
        name="long_60d",
        horizon=60,
        threshold=0.03,
        data_period="10y",
        data_interval="1d",
        min_train_samples=500,
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=25,
        learning_rate=0.03,
        l2_reg=0.5,
    ),
    "long_252d": HorizonConfig(
        name="long_252d",
        horizon=252,
        threshold=0.05,
        data_period="max",
        data_interval="1d",
        min_train_samples=500,
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        learning_rate=0.03,
        l2_reg=0.5,
    ),
}


class HorizonModel:
    """A model optimized for a specific prediction horizon."""

    def __init__(self, config: HorizonConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_trained = False
        self._training_stats: dict = {}

        if config.classifier_kind == "extra_trees":
            self.classifier = ExtraTreesClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_leaf=config.min_samples_leaf,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=model_worker_count(),
            )
        else:
            self.classifier = VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(
                        n_estimators=config.n_estimators,
                        max_depth=config.max_depth,
                        min_samples_leaf=config.min_samples_leaf,
                        max_features="sqrt",
                        random_state=42, n_jobs=model_worker_count(),
                    )),
                    ("hgb", accelerated_classifier(
                        HistGradientBoostingClassifier(
                            max_iter=config.n_estimators,
                            max_depth=config.max_depth,
                            min_samples_leaf=config.min_samples_leaf,
                            learning_rate=config.learning_rate,
                            l2_regularization=config.l2_reg,
                            max_bins=128,
                            early_stopping=True,
                            validation_fraction=0.1,
                            random_state=42,
                        ),
                        n_estimators=config.n_estimators,
                        max_depth=config.max_depth,
                        learning_rate=config.learning_rate,
                        min_child_weight=config.min_samples_leaf,
                        reg_lambda=config.l2_reg,
                        random_state=42,
                    )),
                    ("mlp", MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        max_iter=500,
                        early_stopping=True,
                        random_state=42,
                    )),
                ],
                voting="soft",
                n_jobs=model_worker_count(),
            )

        self.regressor = accelerated_regressor(
            HistGradientBoostingRegressor(
                max_iter=config.n_estimators,
                max_depth=max(2, config.max_depth - 2),
                min_samples_leaf=config.min_samples_leaf,
                learning_rate=config.learning_rate,
                l2_regularization=config.l2_reg,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ),
            n_estimators=config.n_estimators,
            max_depth=max(2, config.max_depth - 2),
            learning_rate=config.learning_rate,
            min_child_weight=config.min_samples_leaf,
            reg_lambda=config.l2_reg,
            random_state=42,
        )

    def prepare_data(
        self, tickers: list[str], context: dict | None = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Download and prepare training data for this horizon."""
        all_features, all_cls, all_reg, all_dates = [], [], [], []

        def prepare_ticker(ticker: str):
            try:
                data = yfinance.Ticker(ticker).history(
                    period=self.config.data_period,
                    interval=self.config.data_interval,
                )
                if data.empty or len(data) < self.config.min_train_samples:
                    return None

                if self.config.use_realtime_features:
                    if not context or "SPY" not in context:
                        raise ValueError("SPY context is required")
                    features = compute_realtime_features(
                        data,
                        context["SPY"],
                    )
                elif self.config.use_intraday_features:
                    features = compute_intraday_features(data, ticker=ticker)
                else:
                    features = compute_features(
                        data, ticker=ticker,
                        include_cross_asset=self.config.use_cross_asset,
                        context=context,
                    )

                if self.config.data_interval.endswith(("m", "h")):
                    cls_target, reg_target = create_intraday_targets(
                        data,
                        horizon=self.config.horizon,
                        threshold=self.config.threshold,
                    )
                else:
                    cls_target = create_target(
                        data,
                        horizon=self.config.horizon,
                        threshold=self.config.threshold,
                    )
                    reg_target = create_regression_target(
                        data,
                        horizon=self.config.horizon,
                    )

                common = (
                    features.index
                    .intersection(cls_target.dropna().index)
                    .intersection(reg_target.dropna().index)
                )
                if len(common) < 100:
                    return None

                features = features.loc[common].replace([np.inf, -np.inf], np.nan).dropna()
                common = features.index.intersection(cls_target.index).intersection(reg_target.index)
                features = features.loc[common]
                cls_target = cls_target.loc[common]
                reg_target = reg_target.loc[common]

                return features, cls_target, reg_target

            except Exception as e:
                logger.debug("Error preparing %s: %s", ticker, e)
                return None

        for prepared in parallel_map(prepare_ticker, tickers):
            if prepared is None:
                continue
            features, cls_target, reg_target = prepared
            all_features.append(features)
            all_cls.append(cls_target)
            all_reg.append(reg_target)
            all_dates.append(pd.Series(features.index))

        if not all_features:
            raise ValueError(f"No valid data for {self.config.name}")

        X = pd.concat(all_features, ignore_index=True)
        y_cls = pd.concat(all_cls, ignore_index=True)
        y_reg = pd.concat(all_reg, ignore_index=True)
        dates = pd.concat(all_dates, ignore_index=True)

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

    def train(self, tickers: list[str], context: dict | None = None) -> dict:
        """Train this horizon model."""
        logger.info("Training %s model...", self.config.name)
        X, y_cls, y_reg = self.prepare_data(tickers, context)
        self.feature_names = list(X.columns)

        logger.info("  %s: %d samples, %d features", self.config.name, len(X), len(self.feature_names))

        X_scaled = self.scaler.fit_transform(X)

        # Time-series CV
        gap = self.config.horizon * max(1, len(tickers))
        test_size = (len(X_scaled) - gap) // 6
        if test_size < 1:
            raise ValueError(
                f"Not enough samples for purged CV: {len(X_scaled)} "
                f"samples with a {gap}-row gap"
            )
        tscv = TimeSeriesSplit(
            n_splits=5,
            gap=gap,
            test_size=test_size,
        )
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_cls.iloc[train_idx], y_cls.iloc[val_idx]
            self.classifier.fit(X_train, y_train)
            score = accuracy_score(y_val, self.classifier.predict(X_val))
            cv_scores.append(score)

        # Final fit
        self.classifier.fit(X_scaled, y_cls)
        if self.config.target_price_mode == "regression":
            self.regressor.fit(X_scaled, y_reg)
        self.is_trained = True

        self._training_stats = {
            "horizon": self.config.name,
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "cv_scores": cv_scores,
            "class_distribution": dict(y_cls.value_counts()),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "compute_backend": compute_backend(),
        }

        logger.info(
            "  %s CV accuracy: %.3f +/- %.3f",
            self.config.name,
            self._training_stats["cv_accuracy_mean"],
            self._training_stats["cv_accuracy_std"],
        )
        return self._training_stats

    def predict(self, X_scaled: np.ndarray) -> tuple[int, float, float]:
        """Predict direction, return, confidence for scaled features."""
        direction = int(self.classifier.predict(X_scaled)[0])
        proba = self.classifier.predict_proba(X_scaled)[0]
        confidence = float(max(proba))
        if self.config.target_price_mode == "threshold":
            pred_return = direction * self.config.threshold
        else:
            pred_return = float(self.regressor.predict(X_scaled)[0])
        return direction, pred_return, confidence

    def predict_ticker(self, ticker: str) -> dict:
        """Generate and persist a current prediction for one ticker."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")

        data = yfinance.Ticker(ticker).history(
            period=self.config.data_period,
            interval=self.config.data_interval,
        )
        if data.empty:
            raise ValueError(f"No data available for {ticker}")

        if self.config.use_realtime_features:
            market_data = yfinance.Ticker("SPY").history(
                period=self.config.data_period,
                interval=self.config.data_interval,
            )
            if market_data.empty:
                raise ValueError("No SPY context available")
            features = compute_realtime_features(data, market_data)
        elif self.config.use_intraday_features:
            features = compute_intraday_features(data, ticker=ticker)
        else:
            features = compute_features(
                data,
                ticker=ticker,
                include_cross_asset=self.config.use_cross_asset,
            )
        if features.empty:
            raise ValueError(f"Insufficient data to compute features for {ticker}")

        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(
                f"Current data is missing {len(missing)} trained features"
            )

        input_at = features.index[-1]
        X = features.iloc[[-1]][self.feature_names]
        X_scaled = self.scaler.transform(X)
        class_label, raw_return, confidence = self.predict(X_scaled)
        direction, predicted_return = reconcile_prediction(
            class_label,
            raw_return,
        )
        predicted_return_pct = predicted_return * 100
        base_price = float(data.loc[input_at, "Close"])
        target_at = prediction_target_time(
            input_at,
            self.config.horizon,
            self.config.data_interval,
        )

        store = PredictionStore()
        stored = store.record_prediction(
            model_name=self.config.model_file,
            family="HorizonModel",
            ticker=ticker,
            input_at=input_at,
            target_at=target_at,
            interval=self.config.data_interval,
            horizon=self.config.horizon,
            base_price=base_price,
            predicted_return_pct=predicted_return_pct,
            direction=direction,
            confidence=confidence,
            last_trained_at=self._training_stats.get("trained_at"),
            training_metrics=self._training_stats,
            metadata={"horizon_name": self.config.name},
        )
        statistics = store.model_statistics(self.config.model_file)
        model_stats = statistics[0] if statistics else {}
        return {
            "model_name": self.config.model_file,
            "horizon_name": self.config.name,
            "ticker": ticker.upper(),
            "direction": direction,
            "predicted_return": predicted_return_pct,
            "confidence": confidence,
            "created_at": stored["created_at"],
            "input_at": stored["input_at"],
            "target_at": stored["target_at"],
            "base_price": stored["base_price"],
            "predicted_price": stored["predicted_price"],
            "last_trained_at": model_stats.get("last_trained_at"),
            "forward_accuracy": model_stats.get("forward_accuracy"),
            "settled_predictions": model_stats.get("settled_count", 0),
            "interval": self.config.data_interval,
            "horizon": self.config.horizon,
        }

    def save(self):
        path = MODEL_DIR / f"{self.config.model_file}.pkl"
        atomic_pickle_dump({
            "classifier": self.classifier,
            "regressor": self.regressor,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_stats": self._training_stats,
            "config": self.config,
        }, path)
        PredictionStore().register_model(
            model_name=self.config.model_file,
            family="HorizonModel",
            horizon=self.config.horizon,
            interval=self.config.data_interval,
            last_trained_at=self._training_stats.get("trained_at"),
            training_metrics=self._training_stats,
        )
        logger.info("  Saved %s", path)

    def load(self):
        path = MODEL_DIR / f"{self.config.model_file}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.regressor = data["regressor"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self._training_stats = data["training_stats"]
        self.is_trained = True
        PredictionStore().register_model(
            model_name=self.config.model_file,
            family="HorizonModel",
            horizon=self.config.horizon,
            interval=self.config.data_interval,
            last_trained_at=self._training_stats.get("trained_at"),
            training_metrics=self._training_stats,
        )


def train_all_horizons(
    tickers: list[str],
    horizons: list[str] | None = None,
) -> dict[str, dict]:
    """Train models for all specified horizons.

    Returns dict of {horizon_name: training_stats}.
    """
    if horizons is None:
        horizons = list(HORIZONS.keys())

    contexts: dict[tuple[str, str], dict] = {}
    results = {}
    for name in horizons:
        if name not in HORIZONS:
            logger.warning("Unknown horizon: %s", name)
            continue
        config = HORIZONS[name]
        model = HorizonModel(config)
        try:
            model_context = None
            if config.use_cross_asset:
                context_key = (config.data_period, config.data_interval)
                if context_key not in contexts:
                    logger.info(
                        "Downloading %s %s cross-asset context data...",
                        *context_key,
                    )
                    contexts[context_key] = _download_context(*context_key)
                model_context = contexts[context_key]
            if config.use_realtime_features:
                spy = yfinance.Ticker("SPY").history(
                    period=config.data_period,
                    interval=config.data_interval,
                )
                model_context = {"SPY": spy}
            stats = model.train(tickers, context=model_context)
            model.save()
            results[name] = stats
        except Exception as e:
            logger.error("Failed to train %s: %s", name, e)
            results[name] = {"error": str(e)}

    return results


def chronological_holdout_horizon(
    horizon_name: str,
    tickers: list[str],
    *,
    test_days: int = 2,
) -> dict:
    """Evaluate the deployed policy on final untouched trading days."""
    if test_days < 1:
        raise ValueError("test_days must be positive")
    config = HORIZONS[horizon_name]

    context = None
    if config.use_cross_asset:
        context = _download_context(config.data_period, config.data_interval)
    elif config.use_realtime_features:
        spy = yfinance.Ticker("SPY").history(
            period=config.data_period,
            interval=config.data_interval,
        )
        if spy.empty:
            return {"error": "No SPY context data"}
        context = {"SPY": spy}

    def prepare_ticker(ticker: str):
        try:
            data = yfinance.Ticker(ticker).history(
                period=config.data_period,
                interval=config.data_interval,
            )
            if data.empty or len(data) < config.min_train_samples:
                return None
            if config.use_realtime_features:
                features = compute_realtime_features(data, context["SPY"])
            elif config.use_intraday_features:
                features = compute_intraday_features(data, ticker=ticker)
            else:
                features = compute_features(
                    data,
                    ticker=ticker,
                    include_cross_asset=config.use_cross_asset,
                    context=context,
                )

            if config.data_interval.endswith(("m", "h")):
                cls_target, reg_target = create_intraday_targets(
                    data,
                    config.horizon,
                    config.threshold,
                )
            else:
                cls_target = create_target(
                    data,
                    config.horizon,
                    config.threshold,
                )
                reg_target = create_regression_target(data, config.horizon)

            common = (
                features.index
                .intersection(cls_target.dropna().index)
                .intersection(reg_target.dropna().index)
            )
            features = (
                features.loc[common]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            common = (
                features.index
                .intersection(cls_target.index)
                .intersection(reg_target.index)
            )
            if len(common) < 100:
                return None
            frame = features.loc[common].copy()
            frame["_target"] = cls_target.loc[common].astype(int)
            frame["_target_return"] = reg_target.loc[common]
            frame["_timestamp"] = common
            frame["_ticker"] = ticker
            return frame
        except Exception as exc:
            logger.warning("Holdout preparation failed for %s: %s", ticker, exc)
            return None

    frames = [
        frame
        for frame in parallel_map(prepare_ticker, sorted(set(tickers)))
        if frame is not None
    ]
    if not frames:
        return {"error": "No valid data"}

    combined = pd.concat(frames, ignore_index=True)
    feature_names = [
        column for column in combined.columns if not column.startswith("_")
    ]
    timestamps = pd.to_datetime(combined["_timestamp"])
    trading_dates = timestamps.dt.date
    unique_dates = sorted(set(trading_dates))
    if len(unique_dates) <= test_days:
        return {"error": "Not enough trading days for holdout"}
    cutoff = unique_dates[-test_days]
    train_mask = trading_dates < cutoff
    test_mask = trading_dates >= cutoff

    model = HorizonModel(config)
    model.feature_names = feature_names
    X_train = combined.loc[train_mask, feature_names]
    y_train = combined.loc[train_mask, "_target"]
    y_reg_train = combined.loc[train_mask, "_target_return"]
    X_test = combined.loc[test_mask, feature_names]
    y_test = combined.loc[test_mask, "_target"].to_numpy()
    actual_returns = combined.loc[test_mask, "_target_return"].to_numpy()

    X_train_scaled = model.scaler.fit_transform(X_train)
    X_test_scaled = model.scaler.transform(X_test)
    model.classifier.fit(X_train_scaled, y_train)
    if config.target_price_mode == "regression":
        model.regressor.fit(X_train_scaled, y_reg_train)

    labels = model.classifier.predict(X_test_scaled).astype(int)
    if config.target_price_mode == "threshold":
        predicted_returns = labels.astype(float) * config.threshold
    else:
        raw_returns = model.regressor.predict(X_test_scaled)
        predicted_returns = np.array([
            reconcile_prediction(int(label), float(value))[1]
            for label, value in zip(labels, raw_returns)
        ])

    directions = np.where(labels > 0, "UP", np.where(labels < 0, "DOWN", "FLAT"))
    hits = np.array([
        directional_hit(
            str(direction),
            1 + float(predicted_return),
            1 + float(actual_return),
        )
        for direction, predicted_return, actual_return in zip(
            directions,
            predicted_returns,
            actual_returns,
        )
    ])
    return {
        "horizon": horizon_name,
        "cutoff": str(cutoff),
        "test_days": test_days,
        "train_predictions": int(train_mask.sum()),
        "total_predictions": int(test_mask.sum()),
        "classification_accuracy": float(np.mean(labels == y_test)),
        "lifecycle_hit_accuracy": float(hits.mean()),
        "up_rate": float(np.mean(labels > 0)),
        "down_rate": float(np.mean(labels < 0)),
        "flat_rate": float(np.mean(labels == 0)),
        "feature_count": len(feature_names),
        "ticker_count": int(combined["_ticker"].nunique()),
    }


def walk_forward_horizon(
    horizon_name: str,
    tickers: list[str],
    test_tickers: list[str] | None = None,
    n_splits: int = 5,
) -> dict:
    """Walk-forward backtest for a specific horizon model."""
    from sklearn.metrics import accuracy_score, classification_report

    if test_tickers is None:
        test_tickers = tickers

    config = HORIZONS[horizon_name]

    # Download context once
    context = None
    if config.use_cross_asset:
        context = _download_context(config.data_period, config.data_interval)
    elif config.use_realtime_features:
        spy = yfinance.Ticker("SPY").history(
            period=config.data_period,
            interval=config.data_interval,
        )
        if spy.empty:
            return {"error": "No SPY context data"}
        context = {"SPY": spy}

    # Prepare all data
    ticker_data = {}
    for ticker in set(tickers) | set(test_tickers):
        try:
            data = yfinance.Ticker(ticker).history(
                period=config.data_period, interval=config.data_interval,
            )
            if data.empty or len(data) < config.min_train_samples:
                continue

            if config.use_realtime_features:
                features = compute_realtime_features(data, context["SPY"])
            elif config.use_intraday_features:
                features = compute_intraday_features(data, ticker=ticker)
            else:
                features = compute_features(
                    data, ticker=ticker,
                    include_cross_asset=config.use_cross_asset,
                    context=context,
                )

            if config.data_interval.endswith(("m", "h")):
                cls_target, reg_target = create_intraday_targets(
                    data,
                    config.horizon,
                    config.threshold,
                )
            else:
                cls_target = create_target(
                    data,
                    horizon=config.horizon,
                    threshold=config.threshold,
                )
                reg_target = create_regression_target(
                    data,
                    horizon=config.horizon,
                )

            common = features.index.intersection(cls_target.dropna().index).intersection(reg_target.dropna().index)
            features = features.loc[common].replace([np.inf, -np.inf], np.nan).dropna()
            common = features.index.intersection(cls_target.index).intersection(reg_target.index)

            if len(common) < 100:
                continue

            ticker_data[ticker] = (
                features.loc[common],
                cls_target.loc[common],
                reg_target.loc[common],
            )
        except Exception as e:
            logger.debug("Error with %s: %s", ticker, e)

    if not ticker_data:
        return {"error": "No valid data"}

    all_dates = sorted(set().union(*(d[0].index for d in ticker_data.values())))
    total_dates = len(all_dates)
    min_train = config.min_train_samples
    split_size = (total_dates - min_train) // n_splits

    if split_size < 5:
        return {"error": "Not enough data for splits"}

    all_true, all_pred, all_hits = [], [], []

    for i in range(n_splits):
        train_end_idx = min_train + i * split_size
        test_end_idx = min(train_end_idx + split_size, total_dates)
        if test_end_idx <= train_end_idx:
            continue

        split_date = all_dates[train_end_idx]
        test_end_date = all_dates[test_end_idx - 1]
        purge_date = all_dates[max(0, train_end_idx - config.horizon)]

        # Collect training data from ALL tickers
        train_X, train_y, train_y_reg = [], [], []
        for ticker in tickers:
            if ticker not in ticker_data:
                continue
            feat, cls_t, reg_t = ticker_data[ticker]
            mask = feat.index < purge_date
            if mask.sum() > 50:
                train_X.append(feat[mask])
                train_y.append(cls_t[mask])
                train_y_reg.append(reg_t[mask])

        if not train_X:
            continue

        X_train = pd.concat(train_X, ignore_index=True)
        y_train = pd.concat(train_y, ignore_index=True)
        y_reg_train = pd.concat(train_y_reg, ignore_index=True)

        model = HorizonModel(config)
        model.feature_names = list(X_train.columns)
        model.scaler = StandardScaler()
        X_train_scaled = model.scaler.fit_transform(X_train)
        model.classifier.fit(X_train_scaled, y_train)
        if config.target_price_mode == "regression":
            model.regressor.fit(X_train_scaled, y_reg_train)

        for ticker in test_tickers:
            if ticker not in ticker_data:
                continue
            feat, cls_t, reg_t = ticker_data[ticker]
            test_mask = (feat.index >= split_date) & (feat.index <= test_end_date)
            if test_mask.sum() == 0:
                continue

            X_test = feat[test_mask]
            y_test = cls_t[test_mask]
            y_reg_test = reg_t[test_mask]

            missing = set(model.feature_names) - set(X_test.columns)
            if missing:
                continue
            X_test = X_test[model.feature_names]
            X_test_scaled = model.scaler.transform(X_test)

            preds = model.classifier.predict(X_test_scaled)
            if config.target_price_mode == "threshold":
                predicted_returns = preds.astype(float) * config.threshold
            else:
                raw_returns = model.regressor.predict(X_test_scaled)
                predicted_returns = np.array([
                    reconcile_prediction(int(label), float(value))[1]
                    for label, value in zip(preds, raw_returns)
                ])
            directions = np.where(
                preds > 0,
                "UP",
                np.where(preds < 0, "DOWN", "FLAT"),
            )
            all_hits.extend(
                directional_hit(
                    str(direction),
                    1 + float(predicted_return),
                    1 + float(actual_return),
                )
                for direction, predicted_return, actual_return in zip(
                    directions,
                    predicted_returns,
                    y_reg_test,
                )
            )
            all_true.extend(y_test.values)
            all_pred.extend(preds)

    if not all_true:
        return {"error": "No predictions made"}

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    accuracy = accuracy_score(all_true, all_pred)

    # Directional accuracy (excluding FLAT)
    dir_mask = (all_true != 0) | (all_pred != 0)
    dir_acc = accuracy_score(all_true[dir_mask], all_pred[dir_mask]) if dir_mask.sum() > 0 else 0

    labels = sorted(set(all_true) | set(all_pred))
    label_map = {-1: "DOWN", 0: "FLAT", 1: "UP"}
    target_names = [label_map.get(l, str(l)) for l in labels]
    report = classification_report(
        all_true, all_pred, labels=labels, target_names=target_names, zero_division=0,
    )

    return {
        "horizon": horizon_name,
        "accuracy": float(accuracy),
        "directional_accuracy": float(dir_acc),
        "lifecycle_hit_accuracy": float(np.mean(all_hits)),
        "total_predictions": len(all_true),
        "classification_report": report,
    }
