"""Continuous collection, prediction, forward evaluation, and retraining."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .compute import compute_backend, parallel_map
from .model import MODEL_DIR, StockPredictor
from .multi_horizon import HORIZONS, HorizonModel, train_all_horizons
from .prediction_store import PredictionStore, iso_utc, utc_now


logger = logging.getLogger(__name__)
STATE_PATH = MODEL_DIR / "continuous_training_state.json"


class ContinuousTrainingService:
    """Maintain models until verified live accuracy reaches the configured goal."""

    def __init__(
        self,
        tickers: list[str],
        *,
        prediction_interval_minutes: int = 60,
        retrain_interval_hours: int = 24,
        target_accuracy: float = 0.70,
        minimum_accuracy: float = 0.65,
        minimum_settled_predictions: int = 30,
        enable_collection: bool = True,
        enable_predictions: bool = True,
        enable_retraining: bool = True,
        stop_event: threading.Event | None = None,
        state_path: str | Path = STATE_PATH,
    ):
        if not 0 < minimum_accuracy <= target_accuracy <= 1:
            raise ValueError(
                "Accuracy settings must satisfy 0 < minimum <= target <= 1"
            )
        self.tickers = sorted({ticker.upper() for ticker in tickers})
        self.prediction_interval = timedelta(minutes=prediction_interval_minutes)
        self.retrain_interval = timedelta(hours=retrain_interval_hours)
        self.target_accuracy = target_accuracy
        self.minimum_accuracy = minimum_accuracy
        self.minimum_settled_predictions = minimum_settled_predictions
        self.enable_collection = enable_collection
        self.enable_predictions = enable_predictions
        self.enable_retraining = enable_retraining
        self.stop_event = stop_event or threading.Event()
        self.state_path = Path(state_path)
        self.store = PredictionStore()
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "cycles": 0,
                "training_rounds": 0,
                "last_cycle_at": None,
                "last_retrained_at": None,
                "last_collection": {},
                "best_model": None,
                "best_forward_accuracy": None,
                "target_achieved": False,
                "minimum_achieved": False,
                "compute_backend": compute_backend(),
                "errors": [],
            }
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read continuous training state {self.state_path}: {exc}"
            ) from exc

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def collect_alternative_data(self) -> dict[str, Any]:
        """Collect all configured public and authenticated optional feeds."""
        from .alternative_data import AlternativeDataCollector

        collector = AlternativeDataCollector()
        result = collector.collect_all(self.tickers)
        self.state["last_collection"] = result
        return result

    def settle_outcomes(self) -> dict[str, Any]:
        """Resolve every due prediction before making model decisions."""
        return self.store.settle_due(now=utc_now(), limit=5000)

    def _predict_with_model(self, model, family: str) -> dict[str, Any]:
        failures: list[dict[str, str]] = []

        def predict_ticker(ticker: str):
            try:
                if family == "default":
                    return model.predict(ticker)
                return model.predict_ticker(ticker)
            except Exception as exc:
                return {"ticker": ticker, "error": str(exc)}

        results = parallel_map(predict_ticker, self.tickers)
        successes = 0
        for result in results:
            if isinstance(result, dict) and result.get("error"):
                failures.append(result)
            else:
                successes += 1
        return {"successes": successes, "failures": failures}

    def generate_predictions(self) -> dict[str, Any]:
        """Generate deduplicated predictions from every trained model."""
        results: dict[str, Any] = {}

        default_path = MODEL_DIR / "stock_predictor.pkl"
        if default_path.exists():
            model = StockPredictor()
            model.load()
            results["stock_predictor"] = self._predict_with_model(model, "default")

        for horizon_name, config in HORIZONS.items():
            if not (MODEL_DIR / f"{config.model_file}.pkl").exists():
                continue
            model = HorizonModel(config)
            model.load()
            results[config.model_file] = self._predict_with_model(model, "horizon")

        combined_path = MODEL_DIR / "combined_model.pkl"
        if combined_path.exists():
            from .sentiment.combined_model import CombinedPredictor

            model = CombinedPredictor()
            model.load()

            def predict_combined(ticker: str):
                try:
                    return model.predict(ticker)
                except Exception as exc:
                    return {"ticker": ticker, "error": str(exc)}

            predictions = parallel_map(predict_combined, self.tickers)
            failures = [
                prediction
                for prediction in predictions
                if prediction.get("error")
            ]
            results["combined_model"] = {
                "successes": len(predictions) - len(failures),
                "failures": failures,
            }

        return results

    def evaluate_forward_accuracy(self) -> dict[str, Any]:
        """Evaluate only settled predictions that were made prospectively."""
        models = self.store.model_statistics()
        eligible = [
            model
            for model in models
            if model["settled_count"] >= self.minimum_settled_predictions
            and model["forward_accuracy"] is not None
        ]
        best = max(
            eligible,
            key=lambda model: model["forward_accuracy"],
            default=None,
        )

        if best:
            accuracy = float(best["forward_accuracy"])
            self.state["best_model"] = best["model_name"]
            self.state["best_forward_accuracy"] = accuracy
            self.state["target_achieved"] = accuracy >= self.target_accuracy
            self.state["minimum_achieved"] = accuracy >= self.minimum_accuracy
        else:
            self.state["best_model"] = None
            self.state["best_forward_accuracy"] = None
            self.state["target_achieved"] = False
            self.state["minimum_achieved"] = False

        return {
            "models": models,
            "eligible_models": len(eligible),
            "minimum_samples": self.minimum_settled_predictions,
            "best_model": self.state["best_model"],
            "best_forward_accuracy": self.state["best_forward_accuracy"],
            "target_accuracy": self.target_accuracy,
            "minimum_accuracy": self.minimum_accuracy,
            "target_achieved": self.state["target_achieved"],
            "minimum_achieved": self.state["minimum_achieved"],
        }

    def retraining_due(self, now: datetime | None = None) -> bool:
        """Return whether models are missing or their refresh interval elapsed."""
        required_models = [
            MODEL_DIR / "stock_predictor.pkl",
            MODEL_DIR / "combined_model.pkl",
            *[
                MODEL_DIR / f"{config.model_file}.pkl"
                for config in HORIZONS.values()
            ],
        ]
        if any(not path.exists() for path in required_models):
            return True

        last_value = self.state.get("last_retrained_at")
        if not last_value:
            return True
        last_retrained = datetime.fromisoformat(last_value.replace("Z", "+00:00"))
        return (now or utc_now()) - last_retrained >= self.retrain_interval

    def retrain_all_models(self) -> dict[str, Any]:
        """Retrain every model family with fresh point-in-time data available."""
        results: dict[str, Any] = {}
        failures: list[dict[str, str]] = []

        try:
            predictor = StockPredictor()
            results["stock_predictor"] = predictor.train(
                self.tickers,
                timeline="5y",
            )
            predictor.save()
        except Exception as exc:
            logger.exception("Default model retraining failed")
            failures.append({"model": "stock_predictor", "error": str(exc)})

        horizon_results = train_all_horizons(self.tickers)
        results.update(horizon_results)
        for name, result in horizon_results.items():
            if "error" in result:
                failures.append({"model": name, "error": result["error"]})

        try:
            from .sentiment.combined_model import CombinedPredictor

            combined = CombinedPredictor(use_sentiment=True, use_macro=True)
            results["combined_model"] = combined.train(self.tickers)
            combined.save()
        except Exception as exc:
            logger.exception("Combined model retraining failed")
            failures.append({"model": "combined_model", "error": str(exc)})

        self.state["training_rounds"] = self.state.get("training_rounds", 0) + 1
        self.state["last_retrained_at"] = iso_utc(utc_now())
        self.state["training_failures"] = failures
        return {"models": results, "failures": failures}

    def run_cycle(self, *, force_retrain: bool = False) -> dict[str, Any]:
        """Run one auditable service cycle."""
        cycle_result: dict[str, Any] = {}
        errors: list[dict[str, str]] = []

        operations = [("settlement", self.settle_outcomes)]
        if self.enable_collection:
            operations.append(("collection", self.collect_alternative_data))

        for name, operation in operations:
            try:
                cycle_result[name] = operation()
            except Exception as exc:
                logger.exception("Continuous %s failed", name)
                errors.append({"operation": name, "error": str(exc)})

        if self.enable_retraining and (force_retrain or self.retraining_due()):
            try:
                cycle_result["training"] = self.retrain_all_models()
            except Exception as exc:
                logger.exception("Continuous retraining cycle failed")
                errors.append({"operation": "training", "error": str(exc)})

        if self.enable_predictions:
            try:
                cycle_result["predictions"] = self.generate_predictions()
            except Exception as exc:
                logger.exception("Continuous prediction cycle failed")
                errors.append({"operation": "predictions", "error": str(exc)})

        cycle_result["forward_evaluation"] = self.evaluate_forward_accuracy()
        self.state["cycles"] = self.state.get("cycles", 0) + 1
        self.state["last_cycle_at"] = iso_utc(utc_now())
        self.state["errors"] = errors[-100:]
        self._save_state()
        return cycle_result

    def run_forever(self) -> None:
        """Run until the process receives a shutdown signal."""
        logger.info(
            "Continuous training started: target=%.1f%%, floor=%.1f%%, "
            "minimum outcomes=%d, backend=%s",
            self.target_accuracy * 100,
            self.minimum_accuracy * 100,
            self.minimum_settled_predictions,
            compute_backend(),
        )
        while not self.stop_event.is_set():
            try:
                result = self.run_cycle()
                evaluation = result["forward_evaluation"]
                logger.info(
                    "Forward evaluation: best=%s accuracy=%s target_met=%s",
                    evaluation["best_model"],
                    evaluation["best_forward_accuracy"],
                    evaluation["target_achieved"],
                )
            except Exception:
                logger.exception("Continuous training cycle failed")
            self.stop_event.wait(self.prediction_interval.total_seconds())


def run_continuous_training(
    tickers: list[str],
    *,
    prediction_interval_minutes: int = 60,
    retrain_interval_hours: int = 24,
    target_accuracy: float = 0.70,
    minimum_accuracy: float = 0.65,
    minimum_settled_predictions: int = 30,
    enable_collection: bool = True,
    enable_predictions: bool = True,
    enable_retraining: bool = True,
    stop_event: threading.Event | None = None,
) -> None:
    """Convenience entry point used by the CLI and all-in-one server."""
    ContinuousTrainingService(
        tickers,
        prediction_interval_minutes=prediction_interval_minutes,
        retrain_interval_hours=retrain_interval_hours,
        target_accuracy=target_accuracy,
        minimum_accuracy=minimum_accuracy,
        minimum_settled_predictions=minimum_settled_predictions,
        enable_collection=enable_collection,
        enable_predictions=enable_predictions,
        enable_retraining=enable_retraining,
        stop_event=stop_event,
    ).run_forever()
