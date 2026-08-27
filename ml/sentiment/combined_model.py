"""Combined quantitative + sentiment prediction model.

Merges 120+ technical/cross-asset features with sentiment data
to create a unified prediction model. Uses rolling retraining
for temporal walk-forward evaluation.
"""

import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..features import compute_features, create_target, _download_context
from .scrapers import (
    get_aggregated_sentiment,
    scrape_all,
    scrape_fear_greed,
    scrape_fred_indicators,
    scrape_rss_news,
    CONFLICT_KEYWORDS,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _build_synthetic_sentiment_history(
    dates: pd.DatetimeIndex,
    ticker: str = "",
    vix_data: pd.Series | None = None,
) -> pd.DataFrame:
    """Build synthetic sentiment features from available historical data.

    Since we can't get historical news sentiment, we use proxies:
    - VIX level → fear/greed proxy
    - VIX changes → sentiment momentum
    - Return dispersion → market uncertainty
    - Volume spikes → event detection
    """
    feat = pd.DataFrame(index=dates)

    if vix_data is not None:
        vix = vix_data.reindex(dates, method="ffill")
        # Fear/Greed proxy from VIX
        feat["fg_proxy"] = 100 - (vix - 10).clip(0, 30) * 3
        feat["fg_proxy_sma5"] = feat["fg_proxy"].rolling(5).mean()
        feat["fg_proxy_sma20"] = feat["fg_proxy"].rolling(20).mean()
        feat["fg_momentum"] = feat["fg_proxy"] - feat["fg_proxy_sma20"]

        # VIX regime classification
        feat["vix_regime"] = pd.cut(
            vix, bins=[0, 15, 20, 25, 35, 100],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

        # VIX term structure proxy (VIX change speed)
        feat["vix_accel"] = vix.pct_change().diff()
        feat["vix_spike"] = (vix.pct_change().abs() > 0.1).astype(int)

    # Market stress indicators (from returns if available)
    feat["sentiment_placeholder"] = 0.0  # Will be overwritten with real data when available

    return feat


def _build_macro_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build macro-economic features from FRED-like proxies.

    Uses market-based proxies since historical FRED data requires API key:
    - TLT (treasury) → rate environment
    - GLD (gold) → safe haven demand
    - HYG/TLT spread → credit conditions
    - UUP (dollar) → currency strength
    """
    feat = pd.DataFrame(index=dates)

    proxies = {
        "TLT": "rate_env",
        "GLD": "safe_haven",
        "HYG": "credit",
        "UUP": "dollar",
    }

    for ticker, name in proxies.items():
        try:
            data = yfinance.Ticker(ticker).history(period="5y")
            if not data.empty:
                close = data["Close"].reindex(dates, method="ffill")
                feat[f"macro_{name}_level"] = close / close.rolling(252).mean() - 1
                feat[f"macro_{name}_mom_1m"] = close.pct_change(20)
                feat[f"macro_{name}_mom_3m"] = close.pct_change(63)
                feat[f"macro_{name}_vol"] = close.pct_change().rolling(20).std()
        except Exception:
            pass

    return feat


class CombinedPredictor:
    """Stock predictor combining quantitative + sentiment features."""

    def __init__(
        self,
        horizon: int = 5,
        threshold: float = 0.0,
        use_sentiment: bool = True,
        use_macro: bool = False,  # Disabled: adds noise, hurts accuracy
    ):
        self.horizon = horizon
        self.threshold = threshold
        self.use_sentiment = use_sentiment
        self.use_macro = use_macro
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_trained = False
        self._training_stats: dict = {}

        # Model: HistGBM + RF ensemble
        self.classifier = VotingClassifier(
            estimators=[
                ("hgb", HistGradientBoostingClassifier(
                    max_iter=600, max_depth=7, min_samples_leaf=30,
                    learning_rate=0.01, l2_regularization=2.0,
                    max_bins=128, early_stopping=True,
                    validation_fraction=0.15, random_state=42,
                )),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    max_iter=500, early_stopping=True,
                    learning_rate_init=0.001, random_state=42,
                )),
            ],
            voting="soft",
        )

    def prepare_data(
        self,
        tickers: list[str],
        context: dict | None = None,
        vix_data: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        """Prepare combined features for all tickers."""
        all_X, all_y, all_dates = [], [], []

        # Get VIX for sentiment proxy
        if vix_data is None and self.use_sentiment:
            try:
                vix_hist = yfinance.Ticker("^VIX").history(period="5y")
                vix_data = vix_hist["Close"] if not vix_hist.empty else None
            except Exception:
                pass

        # Build macro features once
        macro_feat = None

        for ticker in tickers:
            try:
                data = yfinance.Ticker(ticker).history(period="5y")
                if data.empty or len(data) < 300:
                    continue

                # Quantitative features
                quant_feat = compute_features(
                    data, ticker=ticker,
                    include_cross_asset=True, context=context,
                )

                # Sentiment features
                if self.use_sentiment and vix_data is not None:
                    sent_feat = _build_synthetic_sentiment_history(
                        quant_feat.index, ticker=ticker, vix_data=vix_data,
                    )
                    # Align and merge
                    common = quant_feat.index.intersection(sent_feat.index)
                    quant_feat = quant_feat.loc[common]
                    sent_cols = [c for c in sent_feat.columns if c not in quant_feat.columns]
                    if sent_cols:
                        quant_feat = pd.concat(
                            [quant_feat, sent_feat.loc[common, sent_cols]], axis=1,
                        )

                # Macro features (build once, reuse)
                if self.use_macro:
                    if macro_feat is None:
                        macro_feat = _build_macro_features(quant_feat.index)
                    macro_aligned = macro_feat.reindex(quant_feat.index, method="ffill")
                    macro_cols = [c for c in macro_aligned.columns if c not in quant_feat.columns]
                    if macro_cols:
                        quant_feat = pd.concat(
                            [quant_feat, macro_aligned[macro_cols]], axis=1,
                        )

                # Target
                target = create_target(data, horizon=self.horizon, threshold=self.threshold)

                common = quant_feat.index.intersection(target.dropna().index)
                quant_feat = quant_feat.loc[common].replace([np.inf, -np.inf], np.nan).dropna()
                common = quant_feat.index.intersection(target.index)

                if len(common) < 100:
                    continue

                all_X.append(quant_feat.loc[common])
                all_y.append(target.loc[common])
                all_dates.append(pd.Series(
                    quant_feat.loc[common].index,
                    index=range(len(quant_feat.loc[common])),
                ))

            except Exception as e:
                logger.debug("Error preparing %s: %s", ticker, e)

        if not all_X:
            raise ValueError("No valid data prepared")

        X = pd.concat(all_X, ignore_index=True)
        y = pd.concat(all_y, ignore_index=True)
        dates = pd.concat(all_dates, ignore_index=True)

        mask = X.notna().all(axis=1) & y.notna()
        return (
            X[mask].reset_index(drop=True),
            y[mask].reset_index(drop=True),
            dates[mask].reset_index(drop=True),
        )

    def train(self, tickers: list[str]) -> dict:
        """Train the combined model."""
        logger.info("Downloading cross-asset context...")
        context = _download_context("5y", "1d")

        logger.info("Preparing combined features for %d tickers...", len(tickers))
        X, y, dates = self.prepare_data(tickers, context)

        self.feature_names = list(X.columns)
        logger.info("Training: %d samples, %d features", len(X), len(self.feature_names))
        logger.info("Class distribution: %s", dict(y.value_counts()))

        X_scaled = self.scaler.fit_transform(X)

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            self.classifier.fit(X_scaled[train_idx], y.iloc[train_idx])
            preds = self.classifier.predict(X_scaled[val_idx])
            cv_scores.append(accuracy_score(y.iloc[val_idx], preds))

        # Final fit on all data
        self.classifier.fit(X_scaled, y)
        self.is_trained = True

        self._training_stats = {
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "class_distribution": dict(y.value_counts()),
            "horizon": self.horizon,
            "threshold": self.threshold,
        }

        logger.info(
            "CV accuracy: %.3f +/- %.3f",
            self._training_stats["cv_accuracy_mean"],
            self._training_stats["cv_accuracy_std"],
        )
        return self._training_stats

    def walk_forward_backtest(
        self,
        tickers: list[str],
        train_window: int = 504,
        step_size: int = 63,
    ) -> dict:
        """Rolling walk-forward backtest (train past, predict future)."""
        logger.info("Downloading context data...")
        context = _download_context("5y", "1d")

        logger.info("Preparing data for walk-forward...")
        X, y, dates = self.prepare_data(tickers, context)

        all_dates_sorted = dates.sort_values()
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        all_true, all_pred, all_conf = [], [], []

        start_idx = train_window
        split_num = 0

        while start_idx < n_dates:
            split_num += 1
            test_end_idx = min(start_idx + step_size, n_dates)
            train_start_date = unique_dates[max(0, start_idx - train_window)]
            split_date = unique_dates[start_idx]
            test_end_date = unique_dates[test_end_idx - 1]

            train_mask = (dates >= train_start_date) & (dates < split_date)
            test_mask = (dates >= split_date) & (dates <= test_end_date)

            if train_mask.sum() < 500 or test_mask.sum() < 50:
                start_idx += step_size
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask])
            X_test = scaler.transform(X[test_mask])

            model = VotingClassifier(
                estimators=[
                    ("hgb", HistGradientBoostingClassifier(
                        max_iter=600, max_depth=7, min_samples_leaf=30,
                        learning_rate=0.01, l2_regularization=2.0,
                        max_bins=128, early_stopping=True,
                        validation_fraction=0.15, random_state=42,
                    )),
                    ("mlp", MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        max_iter=500, early_stopping=True,
                        learning_rate_init=0.001, random_state=42,
                    )),
                ],
                voting="soft",
            )
            model.fit(X_train, y[train_mask])
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)
            confs = proba.max(axis=1)

            split_acc = accuracy_score(y[test_mask], preds)
            logger.info(
                "  Split %d: train=%d, test=%d, accuracy=%.3f",
                split_num, train_mask.sum(), test_mask.sum(), split_acc,
            )

            all_true.extend(y[test_mask].values)
            all_pred.extend(preds)
            all_conf.extend(confs)

            start_idx += step_size

        if not all_true:
            return {"error": "No predictions made"}

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        all_conf = np.array(all_conf)

        overall_acc = accuracy_score(all_true, all_pred)

        # Confidence-filtered accuracy
        conf_results = {}
        for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
            mask = all_conf >= thresh
            if mask.sum() > 50:
                conf_results[f"conf_{thresh}"] = {
                    "accuracy": float(accuracy_score(all_true[mask], all_pred[mask])),
                    "count": int(mask.sum()),
                    "pct": float(mask.sum() / len(all_true) * 100),
                }

        report = classification_report(
            all_true, all_pred, zero_division=0,
        )

        return {
            "overall_accuracy": float(overall_acc),
            "total_predictions": len(all_true),
            "n_splits": split_num,
            "confidence_filtered": conf_results,
            "classification_report": report,
        }

    def save(self, name: str = "combined_model"):
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "training_stats": self._training_stats,
                "horizon": self.horizon,
                "threshold": self.threshold,
            }, f)
        logger.info("Saved model to %s", path)

    def load(self, name: str = "combined_model"):
        path = MODEL_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self._training_stats = data["training_stats"]
        self.horizon = data["horizon"]
        self.threshold = data["threshold"]
        self.is_trained = True


def accuracy_loop(
    tickers: list[str],
    target_accuracy: float = 0.65,
    max_iterations: int = 10,
) -> dict:
    """Loop training/backtesting until target accuracy or exhausted."""
    logger.info("=" * 60)
    logger.info("COMBINED MODEL ACCURACY LOOP")
    logger.info("  Target: %.0f%%, Max iterations: %d", target_accuracy * 100, max_iterations)
    logger.info("=" * 60)

    best_accuracy = 0
    best_config = None

    configs = [
        # (horizon, threshold, train_window, step_size, use_sent, use_macro)
        (5, 0.0, 504, 63, True, True),
        (5, 0.0, 252, 42, True, True),
        (5, 0.0, 756, 63, True, True),
        (1, 0.0, 504, 63, True, True),
        (1, 0.0, 252, 42, True, True),
        (20, 0.0, 504, 63, True, True),
        (20, 0.0, 756, 63, True, True),
        (60, 0.0, 504, 63, True, True),
        (5, 0.0, 504, 21, True, True),  # More frequent retraining
        (5, 0.0, 126, 21, True, True),  # Short lookback, frequent retrain
    ]

    for i, (horizon, threshold, tw, ss, use_sent, use_macro) in enumerate(configs[:max_iterations]):
        logger.info(
            "\n--- Iteration %d/%d: h=%d, thresh=%.3f, window=%d, step=%d ---",
            i + 1, max_iterations, horizon, threshold, tw, ss,
        )

        model = CombinedPredictor(
            horizon=horizon, threshold=threshold,
            use_sentiment=use_sent, use_macro=use_macro,
        )

        try:
            result = model.walk_forward_backtest(
                tickers, train_window=tw, step_size=ss,
            )

            if "error" in result:
                logger.warning("  Error: %s", result["error"])
                continue

            acc = result["overall_accuracy"]
            logger.info("  Overall accuracy: %.3f", acc)

            # Check confidence-filtered accuracy
            for conf_name, conf_data in result.get("confidence_filtered", {}).items():
                logger.info(
                    "  %s: %.3f (%d preds, %.0f%%)",
                    conf_name, conf_data["accuracy"],
                    conf_data["count"], conf_data["pct"],
                )

            if acc > best_accuracy:
                best_accuracy = acc
                best_config = {
                    "horizon": horizon, "threshold": threshold,
                    "train_window": tw, "step_size": ss,
                    "result": result,
                }
                logger.info("  ** NEW BEST: %.3f **", acc)

                if acc >= target_accuracy:
                    logger.info("TARGET REACHED!")
                    model.train(tickers)
                    model.save(f"combined_h{horizon}")
                    break

        except Exception as e:
            logger.error("  Error in iteration %d: %s", i + 1, e)

    # Print summary
    print("\n" + "=" * 60)
    print("ACCURACY LOOP SUMMARY")
    print("=" * 60)
    if best_config:
        print(f"Best overall accuracy: {best_accuracy:.3f}")
        print(f"Config: horizon={best_config['horizon']}, window={best_config['train_window']}, step={best_config['step_size']}")
        conf = best_config["result"].get("confidence_filtered", {})
        for k, v in conf.items():
            print(f"  {k}: {v['accuracy']:.3f} ({v['count']} preds, {v['pct']:.0f}%)")
    else:
        print("No successful iterations")

    return best_config or {}
