"""Runtime compute selection for parallel CPU and optional GPU workloads."""

from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, Iterable, TypeVar

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone


logger = logging.getLogger(__name__)
T = TypeVar("T")
R = TypeVar("R")


def cpu_count() -> int:
    """Return the CPU budget available to this process."""
    detected = os.process_cpu_count() if hasattr(os, "process_cpu_count") else None
    return max(1, detected or os.cpu_count() or 1)


def io_worker_count(item_count: int | None = None) -> int:
    """Choose a bounded worker count for remote data collection."""
    configured = int(os.getenv("ML_IO_WORKERS", "0"))
    workers = configured or min(16, max(4, cpu_count() * 2))
    return min(workers, item_count) if item_count else workers


def parallel_map(
    function: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
) -> list[R]:
    """Map I/O-heavy work concurrently while preserving input order."""
    materialized = list(items)
    if not materialized:
        return []
    workers = max_workers or io_worker_count(len(materialized))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, materialized))


@lru_cache(maxsize=1)
def nvidia_gpu_available() -> bool:
    """Detect an NVIDIA compute device without making it a hard dependency."""
    requested = os.getenv("ML_ACCELERATOR", "auto").lower()
    if requested == "cpu":
        return False
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        return completed.returncode == 0 and bool(completed.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=1)
def gpu_backend_available() -> bool:
    """Return whether both a CUDA device and the optional XGBoost backend exist."""
    if not nvidia_gpu_available():
        return False
    try:
        import xgboost
    except ImportError:
        logger.info(
            "CUDA device detected; install requirements-gpu.txt to accelerate models"
        )
        return False
    build_info = getattr(xgboost, "build_info", lambda: {})()
    if build_info and not build_info.get("USE_CUDA", False):
        logger.warning("Installed XGBoost build does not include CUDA support")
        return False
    try:
        probe = xgboost.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        probe.fit(X, np.array([0.0, 1.0], dtype=np.float32))
        probe.predict(X[:1])
    except Exception as exc:
        logger.warning("XGBoost CUDA probe failed; using CPU: %s", exc)
        return False
    return True


def compute_backend() -> str:
    """Describe the backend selected for model fitting and inference."""
    return "xgboost-cuda" if gpu_backend_available() else f"sklearn-cpu-{cpu_count()}"


class GPUXGBClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible CUDA classifier that supports arbitrary labels."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 7,
        learning_rate: float = 0.03,
        min_child_weight: float = 20.0,
        reg_lambda: float = 2.0,
        random_state: int = 42,
        fallback=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.fallback = fallback

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self.classes_ = np.unique(np.asarray(y))
        encoded = np.searchsorted(self.classes_, np.asarray(y))
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            eval_metric="logloss",
            n_jobs=cpu_count(),
            random_state=self.random_state,
        )
        try:
            self.model_.fit(X, encoded)
            self.using_fallback_ = False
        except Exception as exc:
            if self.fallback is None:
                raise
            logger.warning("CUDA classifier failed; retrying on CPU: %s", exc)
            self.model_ = clone(self.fallback)
            self.model_.fit(X, y)
            self.using_fallback_ = True
        return self

    def predict(self, X):
        predicted = self.model_.predict(X)
        if self.using_fallback_:
            return predicted
        return self.classes_[predicted.astype(int)]

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class GPUXGBRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible XGBoost CUDA regressor."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.03,
        min_child_weight: float = 20.0,
        reg_lambda: float = 2.0,
        random_state: int = 42,
        fallback=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.fallback = fallback

    def fit(self, X, y):
        from xgboost import XGBRegressor

        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            objective="reg:squarederror",
            n_jobs=cpu_count(),
            random_state=self.random_state,
        )
        try:
            self.model_.fit(X, y)
            self.using_fallback_ = False
        except Exception as exc:
            if self.fallback is None:
                raise
            logger.warning("CUDA regressor failed; retrying on CPU: %s", exc)
            self.model_ = clone(self.fallback)
            self.model_.fit(X, y)
            self.using_fallback_ = True
        return self

    def predict(self, X):
        return self.model_.predict(X)


def accelerated_classifier(fallback, **parameters):
    """Use a CUDA estimator when available, otherwise return the fallback."""
    if not gpu_backend_available():
        return fallback
    return GPUXGBClassifier(fallback=fallback, **parameters)


def accelerated_regressor(fallback, **parameters):
    """Use a CUDA estimator when available, otherwise return the fallback."""
    if not gpu_backend_available():
        return fallback
    return GPUXGBRegressor(fallback=fallback, **parameters)
