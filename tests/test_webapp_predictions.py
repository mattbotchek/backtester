import asyncio
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ml.prediction_store import PredictionStore
from webapp import app as webapp


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name) / "predictions.db"
        self.environment = patch.dict(
            os.environ,
            {"PREDICTION_DB_PATH": str(self.database)},
        )
        self.environment.start()
        webapp._last_settlement_at = None
        webapp._settlement_in_progress = False
        self.store = PredictionStore(self.database)
        self.prediction = self.store.record_prediction(
            model_name="model_short_1d",
            family="HorizonModel",
            ticker="AAPL",
            input_at="2099-01-05T20:00:00Z",
            target_at="2099-01-06T20:00:00Z",
            interval="1d",
            horizon=1,
            base_price=100,
            predicted_return_pct=2,
            direction="UP",
            confidence=0.75,
            last_trained_at="2099-01-05T19:00:00Z",
        )

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def test_history_and_model_status_expose_forward_metadata(self):
        history_response = asyncio.run(
            webapp.api_prediction_history(ticker="AAPL", model=None, limit=200)
        )
        status_response = asyncio.run(webapp.api_model_status())

        history = history_response["predictions"]
        self.assertEqual(1, len(history))
        self.assertEqual(self.prediction["target_at"], history[0]["target_at"])
        self.assertEqual("pending", history[0]["status"])
        self.assertIn("actual_price", history[0])

        model = status_response["models"][0]
        self.assertEqual("2099-01-05T19:00:00Z", model["last_trained_at"])
        self.assertEqual(1, model["prediction_count"])
        self.assertEqual(0, model["settled_count"])
        self.assertIsNone(model["forward_accuracy"])

    def test_stock_and_model_pages_include_history_and_overlay_surfaces(self):
        templates = Path(webapp.__file__).parent / "templates"
        stock_page = (templates / "stock_detail.html").read_text(encoding="utf-8")
        model_page = (templates / "ml_control.html").read_text(encoding="utf-8")

        self.assertIn("Prediction History", stock_page)
        self.assertIn("predicted_price", stock_page)
        self.assertIn("actual_price", stock_page)
        self.assertIn("Last trained", stock_page)

        self.assertIn("Live Forward Accuracy", model_page)
        self.assertIn("/api/predictions/history", model_page)

    def test_startup_auto_training_can_be_disabled(self):
        with (
            patch.dict(os.environ, {"WEBAPP_AUTO_TRAIN": "0"}),
            patch.object(webapp.threading, "Thread") as thread,
        ):
            asyncio.run(webapp.on_startup())

        thread.assert_not_called()

    def test_endpoint_settlement_runs_network_work_in_background(self):
        entered = threading.Event()
        release = threading.Event()

        def slow_settlement(store, *, now):
            entered.set()
            release.wait(timeout=2)
            return {"settled": 0}

        with patch.object(
            webapp.PredictionStore,
            "settle_due",
            autospec=True,
            side_effect=slow_settlement,
        ):
            first = webapp._settle_due_predictions()
            self.assertTrue(entered.wait(timeout=1))
            second = webapp._settle_due_predictions()
            release.set()
            for _ in range(100):
                if not webapp._settlement_in_progress:
                    break
                time.sleep(0.01)

        self.assertEqual("started", first["status"])
        self.assertEqual("in_progress", second["status"])
        self.assertFalse(webapp._settlement_in_progress)

    def test_intraday_bars_use_absolute_utc_timestamps(self):
        self.store.record_prediction(
            model_name="model_realtime_5m",
            family="HorizonModel",
            ticker="AAPL",
            input_at="2099-01-05T20:00:00Z",
            target_at="2099-01-05T20:05:00Z",
            interval="1m",
            horizon=5,
            base_price=100,
            predicted_return_pct=0,
            direction="FLAT",
            confidence=0.75,
            last_trained_at="2099-01-05T19:00:00Z",
        )
        index = pd.DatetimeIndex(
            ["2026-08-28 09:30:00"],
            tz="America/New_York",
        )
        history = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1000],
            },
            index=index,
        )
        ticker = unittest.mock.Mock()
        ticker.history.return_value = history

        with patch.object(webapp.yfinance, "Ticker", return_value=ticker):
            response = asyncio.run(
                webapp.api_intraday("AAPL", period="1d", interval="5m")
            )

        self.assertEqual("2026-08-28T13:30:00Z", response["bars"][0]["time"])
        self.assertEqual(
            ["model_realtime_5m"],
            [item["model_name"] for item in response["predictions"]],
        )


if __name__ == "__main__":
    unittest.main()
