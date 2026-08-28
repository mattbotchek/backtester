import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ml.continuous_training import ContinuousTrainingService
from ml.prediction_store import PredictionStore


class ContinuousTrainingTests(unittest.TestCase):
    def test_forward_gate_uses_settled_live_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "predictions.db"
            state = Path(directory) / "state.json"
            with patch.dict(
                os.environ,
                {"PREDICTION_DB_PATH": str(database)},
            ):
                service = ContinuousTrainingService(
                    ["AAPL"],
                    minimum_settled_predictions=4,
                    state_path=state,
                )
                store = PredictionStore(database)
                start = datetime(2026, 1, 1, tzinfo=timezone.utc)
                for index, actual in enumerate((102.0, 103.0, 101.0, 99.0)):
                    prediction = store.record_prediction(
                        model_name="candidate",
                        family="test",
                        ticker=f"T{index}",
                        input_at=start + timedelta(days=index),
                        target_at=start + timedelta(days=index + 1),
                        interval="1d",
                        horizon=1,
                        base_price=100.0,
                        predicted_return_pct=1.0,
                        direction="UP",
                        confidence=0.8,
                    )
                    store.settle_prediction(
                        prediction["id"],
                        actual,
                        start + timedelta(days=index + 1),
                    )

                evaluation = service.evaluate_forward_accuracy()

                self.assertEqual("candidate", evaluation["best_model"])
                self.assertEqual(0.75, evaluation["best_forward_accuracy"])
                self.assertTrue(evaluation["target_achieved"])
                self.assertTrue(evaluation["minimum_achieved"])

    def test_pending_predictions_do_not_satisfy_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "predictions.db"
            with patch.dict(
                os.environ,
                {"PREDICTION_DB_PATH": str(database)},
            ):
                service = ContinuousTrainingService(
                    ["AAPL"],
                    minimum_settled_predictions=1,
                    state_path=Path(directory) / "state.json",
                )
                service.store.record_prediction(
                    model_name="candidate",
                    family="test",
                    ticker="AAPL",
                    input_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    target_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                    interval="1d",
                    horizon=1,
                    base_price=100.0,
                    predicted_return_pct=1.0,
                    direction="UP",
                    confidence=0.8,
                )

                evaluation = service.evaluate_forward_accuracy()

                self.assertIsNone(evaluation["best_model"])
                self.assertFalse(evaluation["target_achieved"])


if __name__ == "__main__":
    unittest.main()
