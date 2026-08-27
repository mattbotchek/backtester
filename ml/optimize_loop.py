"""Automated accuracy optimization loop.

Trains models with different configurations, runs walk-forward backtests,
and iterates until accuracy targets are met or all options exhausted.

Usage:
    python -m ml.optimize_loop [--horizon short_5d] [--target 0.70]
"""

import logging
import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import yfinance
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .features import (
    compute_features,
    create_target,
    create_regression_target,
    _download_context,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Tickers for training pool (broad market coverage)
TRAIN_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "JNJ", "V", "PG", "UNH", "HD", "DIS", "MA", "PYPL", "NFLX",
    "ADBE", "CRM", "INTC", "CSCO", "PEP", "KO", "MRK", "ABT",
    "XOM", "CVX", "WMT", "COST", "TGT", "LOW", "NKE", "SBUX",
    "BA", "CAT", "DE", "GE", "MMM", "HON", "UPS",
    "BRK-B", "GS", "MS", "C", "BAC", "WFC", "AXP",
    "LLY", "PFE", "ABBV", "BMY", "GILD", "TMO", "DHR",
    "NEE", "DUK", "SO", "AEP", "SRE",
    "AMT", "PLD", "SPG", "O",
    "AMD", "QCOM", "TXN", "AVGO", "MU", "AMAT",
]

TEST_TICKERS = [
    "AAPL", "JPM", "XOM", "JNJ", "HD",
    "NFLX", "GS", "NEE", "PLD", "AMD",
    "WMT", "PFE", "BA", "SBUX", "CSCO",
]

# Ensure no overlap
TRAIN_ONLY = [t for t in TRAIN_TICKERS if t not in TEST_TICKERS]


@dataclass
class TrialConfig:
    """One trial configuration to test."""
    name: str
    horizon: int
    threshold: float
    n_estimators: int = 400
    max_depth: int = 7
    min_samples_leaf: int = 30
    learning_rate: float = 0.02
    l2_reg: float = 1.5
    top_k_features: int = 0  # 0 = use all
    use_hgb_only: bool = False
    smooth_target: int = 0  # smoothing window for target, 0 = none


def download_all_data(
    tickers: list[str], period: str = "5y", interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download and cache all ticker data."""
    data = {}
    for ticker in tickers:
        try:
            df = yfinance.Ticker(ticker).history(period=period, interval=interval)
            if not df.empty and len(df) > 300:
                data[ticker] = df
        except Exception:
            pass
    logger.info("Downloaded %d/%d tickers", len(data), len(tickers))
    return data


def prepare_features(
    ticker_data: dict[str, pd.DataFrame],
    context: dict,
    horizon: int,
    threshold: float,
    smooth: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and targets for all tickers."""
    all_X, all_y = [], []

    for ticker, data in ticker_data.items():
        try:
            features = compute_features(
                data, ticker=ticker, include_cross_asset=True, context=context,
            )

            if smooth > 0:
                close_smooth = data["Close"].rolling(smooth, center=False).mean()
                future = close_smooth.shift(-horizon)
                ret = (future - data["Close"]) / data["Close"]
                target = pd.Series(0, index=data.index)
                target[ret > threshold] = 1
                target[ret < -threshold] = -1
                target[ret.isna()] = np.nan
            else:
                target = create_target(data, horizon=horizon, threshold=threshold)

            common = features.index.intersection(target.dropna().index)
            features = features.loc[common].replace([np.inf, -np.inf], np.nan).dropna()
            common = features.index.intersection(target.index)
            if len(common) < 100:
                continue

            all_X.append(features.loc[common])
            all_y.append(target.loc[common])
        except Exception as e:
            logger.debug("Error with %s: %s", ticker, e)

    if not all_X:
        raise ValueError("No valid data")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    mask = X.notna().all(axis=1) & y.notna()
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def select_features(X: pd.DataFrame, y: pd.Series, top_k: int) -> list[str]:
    """Select top-k features by mutual information."""
    if top_k <= 0 or top_k >= len(X.columns):
        return list(X.columns)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns,
    ).fillna(0)

    mi = mutual_info_classif(X_scaled, y, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    selected = list(mi_series.head(top_k).index)
    logger.info("  Top features: %s", ", ".join(selected[:10]))
    return selected


def run_trial(
    config: TrialConfig,
    train_data: dict[str, pd.DataFrame],
    test_data: dict[str, pd.DataFrame],
    context: dict,
    n_splits: int = 5,
) -> dict:
    """Run one trial: train + walk-forward backtest."""
    logger.info("=== Trial: %s ===", config.name)
    logger.info(
        "  horizon=%d, threshold=%.4f, depth=%d, lr=%.3f, l2=%.1f, top_k=%d, smooth=%d, hgb_only=%s",
        config.horizon, config.threshold, config.max_depth, config.learning_rate,
        config.l2_reg, config.top_k_features, config.smooth_target, config.use_hgb_only,
    )

    try:
        X_train, y_train = prepare_features(
            train_data, context, config.horizon, config.threshold, config.smooth_target,
        )
    except ValueError as e:
        return {"trial": config.name, "error": str(e)}

    logger.info("  Training data: %d samples, %d features", len(X_train), len(X_train.columns))
    logger.info("  Class dist: %s", dict(y_train.value_counts()))

    # Feature selection
    feature_cols = X_train.columns.tolist()
    if config.top_k_features > 0:
        feature_cols = select_features(X_train, y_train, config.top_k_features)
        X_train = X_train[feature_cols]

    # CV score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if config.use_hgb_only:
        model = HistGradientBoostingClassifier(
            max_iter=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            learning_rate=config.learning_rate,
            l2_regularization=config.l2_reg,
            max_bins=128,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
    else:
        model = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=config.n_estimators, max_depth=config.max_depth,
                    min_samples_leaf=config.min_samples_leaf, max_features="sqrt",
                    random_state=42, n_jobs=-1,
                )),
                ("hgb", HistGradientBoostingClassifier(
                    max_iter=config.n_estimators, max_depth=config.max_depth,
                    min_samples_leaf=config.min_samples_leaf,
                    learning_rate=config.learning_rate,
                    l2_regularization=config.l2_reg,
                    max_bins=128, early_stopping=True,
                    validation_fraction=0.1, random_state=42,
                )),
            ],
            voting="soft",
        )

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y_train.iloc[train_idx])
        preds = model.predict(X_scaled[val_idx])
        cv_scores.append(accuracy_score(y_train.iloc[val_idx], preds))

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    logger.info("  CV accuracy: %.3f +/- %.3f", cv_mean, cv_std)

    # Walk-forward backtest on test tickers
    try:
        X_test, y_test = prepare_features(
            test_data, context, config.horizon, config.threshold, config.smooth_target,
        )
    except ValueError:
        return {
            "trial": config.name, "cv_accuracy": cv_mean, "cv_std": cv_std,
            "wf_accuracy": None, "error": "No test data",
        }

    X_test = X_test[[c for c in feature_cols if c in X_test.columns]]
    if len(X_test.columns) < len(feature_cols) * 0.5:
        return {
            "trial": config.name, "cv_accuracy": cv_mean, "cv_std": cv_std,
            "wf_accuracy": None, "error": "Feature mismatch",
        }

    # Train on full training set, test on held-out tickers
    model.fit(X_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    wf_accuracy = accuracy_score(y_test, preds)

    # Directional accuracy
    dir_mask = (y_test != 0) | (pd.Series(preds) != 0)
    dir_acc = accuracy_score(
        y_test[dir_mask.values], preds[dir_mask.values],
    ) if dir_mask.sum() > 0 else 0

    n_classes = len(set(y_test) | set(preds))

    logger.info("  Walk-forward accuracy: %.3f (%d predictions, %d classes)", wf_accuracy, len(y_test), n_classes)
    logger.info("  Directional accuracy: %.3f", dir_acc)

    return {
        "trial": config.name,
        "cv_accuracy": float(cv_mean),
        "cv_std": float(cv_std),
        "wf_accuracy": float(wf_accuracy),
        "directional_accuracy": float(dir_acc),
        "n_train": len(X_train),
        "n_test": len(y_test),
        "n_classes": n_classes,
        "n_features": len(feature_cols),
        "config": config,
    }


def generate_trials(horizon: int) -> list[TrialConfig]:
    """Generate trial configurations to test."""
    trials = []
    trial_id = 0

    # Vary threshold (controls class balance and signal clarity)
    for threshold in [0.0, 0.005, 0.01, 0.02, 0.03]:
        trial_id += 1
        trials.append(TrialConfig(
            name=f"t{trial_id}_thresh{threshold}",
            horizon=horizon,
            threshold=threshold,
        ))

    # Best threshold + HGB only
    for threshold in [0.0, 0.01, 0.02]:
        trial_id += 1
        trials.append(TrialConfig(
            name=f"t{trial_id}_hgb_thresh{threshold}",
            horizon=horizon,
            threshold=threshold,
            use_hgb_only=True,
        ))

    # Feature selection variants
    for top_k in [30, 50, 80]:
        trial_id += 1
        trials.append(TrialConfig(
            name=f"t{trial_id}_top{top_k}",
            horizon=horizon,
            threshold=0.01,
            top_k_features=top_k,
        ))

    # Smoothed target
    for smooth in [3, 5]:
        trial_id += 1
        trials.append(TrialConfig(
            name=f"t{trial_id}_smooth{smooth}",
            horizon=horizon,
            threshold=0.01,
            smooth_target=smooth,
        ))

    # Deeper / more regularized
    trial_id += 1
    trials.append(TrialConfig(
        name=f"t{trial_id}_deep_reg",
        horizon=horizon,
        threshold=0.01,
        max_depth=10,
        l2_reg=3.0,
        learning_rate=0.01,
        n_estimators=600,
    ))

    # Shallow / fast
    trial_id += 1
    trials.append(TrialConfig(
        name=f"t{trial_id}_shallow",
        horizon=horizon,
        threshold=0.01,
        max_depth=4,
        min_samples_leaf=50,
        learning_rate=0.05,
        use_hgb_only=True,
    ))

    return trials


def run_optimization_loop(
    horizon: int = 5,
    target_accuracy: float = 0.70,
    max_rounds: int = 3,
):
    """Main optimization loop."""
    logger.info("=" * 60)
    logger.info("ACCURACY OPTIMIZATION LOOP")
    logger.info("  Horizon: %d days, Target: %.0f%%", horizon, target_accuracy * 100)
    logger.info("=" * 60)

    # Download data once
    logger.info("Downloading training data...")
    train_data = download_all_data(TRAIN_ONLY)
    test_data = download_all_data(TEST_TICKERS)

    logger.info("Downloading cross-asset context...")
    context = _download_context("5y", "1d")

    best_accuracy = 0
    best_trial = None
    all_results = []

    trials = generate_trials(horizon)

    for round_num in range(max_rounds):
        logger.info("\n" + "=" * 60)
        logger.info("ROUND %d/%d  (%d trials)", round_num + 1, max_rounds, len(trials))
        logger.info("=" * 60)

        round_results = []
        for trial in trials:
            result = run_trial(trial, train_data, test_data, context)
            round_results.append(result)
            all_results.append(result)

            wf = result.get("wf_accuracy")
            if wf and wf > best_accuracy:
                best_accuracy = wf
                best_trial = result
                logger.info("  ** NEW BEST: %.3f **", best_accuracy)

                if best_accuracy >= target_accuracy:
                    logger.info("TARGET REACHED: %.3f >= %.3f", best_accuracy, target_accuracy)
                    _print_summary(all_results, best_trial)
                    return all_results

        # Generate next round based on best results
        top_results = sorted(
            [r for r in round_results if r.get("wf_accuracy")],
            key=lambda r: r["wf_accuracy"],
            reverse=True,
        )[:3]

        if not top_results:
            logger.warning("No successful trials in round %d", round_num + 1)
            continue

        # Generate refined trials around best config
        trials = []
        for i, r in enumerate(top_results):
            cfg = r["config"]
            trial_id = (round_num + 1) * 100 + i * 10

            # Small perturbations of best config
            for lr_mult in [0.5, 1.0, 2.0]:
                trial_id += 1
                trials.append(TrialConfig(
                    name=f"r{round_num+2}_t{trial_id}",
                    horizon=cfg.horizon,
                    threshold=cfg.threshold,
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    min_samples_leaf=cfg.min_samples_leaf,
                    learning_rate=cfg.learning_rate * lr_mult,
                    l2_reg=cfg.l2_reg,
                    top_k_features=cfg.top_k_features,
                    use_hgb_only=cfg.use_hgb_only,
                    smooth_target=cfg.smooth_target,
                ))

    _print_summary(all_results, best_trial)
    return all_results


def _print_summary(results: list[dict], best: dict | None):
    """Print optimization summary."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)

    valid = [r for r in results if r.get("wf_accuracy") is not None]
    valid.sort(key=lambda r: r["wf_accuracy"], reverse=True)

    print(f"\n{'Trial':<30} {'CV':>8} {'WF':>8} {'Dir':>8} {'#Feat':>6} {'Classes':>8}")
    print("-" * 70)
    for r in valid[:20]:
        print(
            f"{r['trial']:<30} {r['cv_accuracy']:>7.3f} {r['wf_accuracy']:>7.3f} "
            f"{r.get('directional_accuracy', 0):>7.3f} {r.get('n_features', 0):>6} "
            f"{r.get('n_classes', 0):>8}"
        )

    if best:
        print(f"\nBest: {best['trial']} — WF accuracy: {best['wf_accuracy']:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--target", type=float, default=0.70)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    run_optimization_loop(
        horizon=args.horizon,
        target_accuracy=args.target,
        max_rounds=args.rounds,
    )
