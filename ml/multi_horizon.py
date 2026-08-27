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
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
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
    create_target,
    create_regression_target,
    _download_context,
)

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

    @property
    def model_file(self) -> str:
        return f"model_{self.name}"


# Pre-configured horizons
HORIZONS = {
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
        data_interval="1wk",
        min_train_samples=200,
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

        self.classifier = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    min_samples_leaf=config.min_samples_leaf,
                    max_features="sqrt",
                    random_state=42, n_jobs=-1,
                )),
                ("hgb", HistGradientBoostingClassifier(
                    max_iter=config.n_estimators,
                    max_depth=config.max_depth,
                    min_samples_leaf=config.min_samples_leaf,
                    learning_rate=config.learning_rate,
                    l2_regularization=config.l2_reg,
                    max_bins=128,
                    early_stopping=True,
                    validation_fraction=0.1,
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
        )

        self.regressor = GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth - 2,
            min_samples_leaf=config.min_samples_leaf,
            learning_rate=config.learning_rate,
            subsample=0.8,
            random_state=42,
        )

    def prepare_data(
        self, tickers: list[str], context: dict | None = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Download and prepare training data for this horizon."""
        all_features, all_cls, all_reg = [], [], []

        for ticker in tickers:
            try:
                data = yfinance.Ticker(ticker).history(
                    period=self.config.data_period,
                    interval=self.config.data_interval,
                )
                if data.empty or len(data) < self.config.min_train_samples:
                    continue

                if self.config.use_intraday_features:
                    features = compute_intraday_features(data)
                else:
                    features = compute_features(
                        data, ticker=ticker,
                        include_cross_asset=self.config.use_cross_asset,
                        context=context,
                    )

                cls_target = create_target(
                    data, horizon=self.config.horizon,
                    threshold=self.config.threshold,
                )
                reg_target = create_regression_target(
                    data, horizon=self.config.horizon,
                )

                common = (
                    features.index
                    .intersection(cls_target.dropna().index)
                    .intersection(reg_target.dropna().index)
                )
                if len(common) < 100:
                    continue

                features = features.loc[common].replace([np.inf, -np.inf], np.nan).dropna()
                common = features.index.intersection(cls_target.index).intersection(reg_target.index)
                features = features.loc[common]
                cls_target = cls_target.loc[common]
                reg_target = reg_target.loc[common]

                all_features.append(features)
                all_cls.append(cls_target)
                all_reg.append(reg_target)

            except Exception as e:
                logger.debug("Error preparing %s: %s", ticker, e)

        if not all_features:
            raise ValueError(f"No valid data for {self.config.name}")

        X = pd.concat(all_features, ignore_index=True)
        y_cls = pd.concat(all_cls, ignore_index=True)
        y_reg = pd.concat(all_reg, ignore_index=True)

        mask = X.notna().all(axis=1) & y_cls.notna() & y_reg.notna()
        X = X[mask].reset_index(drop=True)
        y_cls = y_cls[mask].reset_index(drop=True)
        y_reg = y_reg[mask].reset_index(drop=True)

        return X, y_cls, y_reg

    def train(self, tickers: list[str], context: dict | None = None) -> dict:
        """Train this horizon model."""
        logger.info("Training %s model...", self.config.name)
        X, y_cls, y_reg = self.prepare_data(tickers, context)
        self.feature_names = list(X.columns)

        logger.info("  %s: %d samples, %d features", self.config.name, len(X), len(self.feature_names))

        X_scaled = self.scaler.fit_transform(X)

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_cls.iloc[train_idx], y_cls.iloc[val_idx]
            self.classifier.fit(X_train, y_train)
            score = accuracy_score(y_val, self.classifier.predict(X_val))
            cv_scores.append(score)

        # Final fit
        self.classifier.fit(X_scaled, y_cls)
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
        pred_return = float(self.regressor.predict(X_scaled)[0])
        return direction, pred_return, confidence

    def save(self):
        path = MODEL_DIR / f"{self.config.model_file}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "regressor": self.regressor,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "training_stats": self._training_stats,
                "config": self.config,
            }, f)
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


def train_all_horizons(
    tickers: list[str],
    horizons: list[str] | None = None,
) -> dict[str, dict]:
    """Train models for all specified horizons.

    Returns dict of {horizon_name: training_stats}.
    """
    if horizons is None:
        horizons = list(HORIZONS.keys())

    # Download cross-asset context once
    logger.info("Downloading cross-asset context data...")
    context = _download_context("5y", "1d")
    # Flatten if nested
    if len(context) == 1:
        context = list(context.values())[0]

    results = {}
    for name in horizons:
        if name not in HORIZONS:
            logger.warning("Unknown horizon: %s", name)
            continue
        config = HORIZONS[name]
        model = HorizonModel(config)
        try:
            stats = model.train(tickers, context=context if config.use_cross_asset else None)
            model.save()
            results[name] = stats
        except Exception as e:
            logger.error("Failed to train %s: %s", name, e)
            results[name] = {"error": str(e)}

    return results


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
        if len(context) == 1:
            context = list(context.values())[0]

    # Prepare all data
    ticker_data = {}
    for ticker in set(tickers) | set(test_tickers):
        try:
            data = yfinance.Ticker(ticker).history(
                period=config.data_period, interval=config.data_interval,
            )
            if data.empty or len(data) < config.min_train_samples:
                continue

            if config.use_intraday_features:
                features = compute_intraday_features(data)
            else:
                features = compute_features(
                    data, ticker=ticker,
                    include_cross_asset=config.use_cross_asset,
                    context=context,
                )

            cls_target = create_target(data, horizon=config.horizon, threshold=config.threshold)
            reg_target = create_regression_target(data, horizon=config.horizon)

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

    all_true, all_pred = [], []

    for i in range(n_splits):
        train_end_idx = min_train + i * split_size
        test_end_idx = min(train_end_idx + split_size, total_dates)
        if test_end_idx <= train_end_idx:
            continue

        split_date = all_dates[train_end_idx]
        test_end_date = all_dates[test_end_idx - 1]

        # Collect training data from ALL tickers
        train_X, train_y = [], []
        for ticker in tickers:
            if ticker not in ticker_data:
                continue
            feat, cls_t, _ = ticker_data[ticker]
            mask = feat.index < split_date
            if mask.sum() > 50:
                train_X.append(feat[mask])
                train_y.append(cls_t[mask])

        if not train_X:
            continue

        X_train = pd.concat(train_X, ignore_index=True)
        y_train = pd.concat(train_y, ignore_index=True)

        model = HorizonModel(config)
        model.feature_names = list(X_train.columns)
        model.scaler = StandardScaler()
        X_train_scaled = model.scaler.fit_transform(X_train)
        model.classifier.fit(X_train_scaled, y_train)

        for ticker in test_tickers:
            if ticker not in ticker_data:
                continue
            feat, cls_t, _ = ticker_data[ticker]
            test_mask = (feat.index >= split_date) & (feat.index <= test_end_date)
            if test_mask.sum() == 0:
                continue

            X_test = feat[test_mask]
            y_test = cls_t[test_mask]

            missing = set(model.feature_names) - set(X_test.columns)
            if missing:
                continue
            X_test = X_test[model.feature_names]
            X_test_scaled = model.scaler.transform(X_test)

            preds = model.classifier.predict(X_test_scaled)
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
        "total_predictions": len(all_true),
        "classification_report": report,
    }
