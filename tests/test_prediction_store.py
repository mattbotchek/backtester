import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ml.prediction_store import (
    PredictionStore,
    directional_hit,
    prediction_target_time,
)


class PredictionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = PredictionStore(
            Path(self.temporary_directory.name) / "predictions.db"
        )
        self.input_at = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)
        self.target_at = datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def record(self, **overrides):
        values = {
            "model_name": "model_short_1d",
            "family": "HorizonModel",
            "ticker": "AAPL",
            "input_at": self.input_at,
            "target_at": self.target_at,
            "interval": "1d",
            "horizon": 1,
            "base_price": 100.0,
            "predicted_return_pct": 2.0,
            "direction": "UP",
            "confidence": 0.72,
            "last_trained_at": self.input_at,
            "training_metrics": {"score": 0.61},
        }
        values.update(overrides)
        return self.store.record_prediction(**values)

    def test_prediction_is_idempotent_for_same_model_input_and_target(self):
        first = self.record()
        second = self.record(confidence=0.99)

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(0.72, second["confidence"])
        self.assertEqual(1, len(self.store.get_history()))

    def test_settlement_uses_predicted_target_price_rule(self):
        prediction = self.record()
        settled = self.store.settle_prediction(
            prediction["id"],
            actual_price=102.0,
            actual_at=self.target_at,
        )

        self.assertTrue(settled["hit"])
        self.assertEqual("settled", settled["status"])
        stats = self.store.model_statistics("model_short_1d")[0]
        self.assertEqual(1, stats["settled_count"])
        self.assertEqual(1.0, stats["forward_accuracy"])

    def test_due_settlement_ignores_observations_before_target(self):
        prediction = self.record()

        result = self.store.settle_due(
            now=datetime(2026, 9, 1, tzinfo=timezone.utc),
            price_fetcher=lambda ticker, target, interval: [
                (datetime(2026, 8, 31, 19, 59, tzinfo=timezone.utc), 110.0),
                (datetime(2026, 8, 31, 20, 1, tzinfo=timezone.utc), 101.0),
            ],
        )

        self.assertEqual(1, result["settled"])
        settled = self.store.get_history()[0]
        self.assertEqual(101.0, settled["actual_price"])
        self.assertFalse(settled["hit"])
        self.assertEqual(prediction["id"], settled["id"])

    def test_history_filters_by_ticker_and_model(self):
        self.record()
        self.record(
            model_name="stock_predictor",
            family="StockPredictor",
            ticker="MSFT",
        )

        aapl = self.store.get_history(ticker="AAPL")
        default = self.store.get_history(model_name="stock_predictor")

        self.assertEqual(["AAPL"], [item["ticker"] for item in aapl])
        self.assertEqual(["MSFT"], [item["ticker"] for item in default])

    def test_business_day_target_and_direction_rules(self):
        friday = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)
        target = prediction_target_time(friday, 1, "1d")

        self.assertEqual(
            datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc),
            target,
        )
        self.assertTrue(directional_hit("UP", 102, 102))
        self.assertTrue(directional_hit("DOWN", 98, 97))
        self.assertFalse(directional_hit("DOWN", 98, 99))

    def test_daily_target_respects_holidays_and_early_closes(self):
        before_independence_day = datetime(
            2026, 7, 2, 20, 0, tzinfo=timezone.utc
        )
        before_thanksgiving = datetime(
            2026, 11, 25, 21, 0, tzinfo=timezone.utc
        )

        self.assertEqual(
            datetime(2026, 7, 6, 20, 0, tzinfo=timezone.utc),
            prediction_target_time(before_independence_day, 1, "1d"),
        )
        self.assertEqual(
            datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc),
            prediction_target_time(before_thanksgiving, 1, "1d"),
        )

    def test_intraday_target_rejects_cross_session_horizons(self):
        friday_at_two_et = datetime(
            2026, 8, 28, 18, 0, tzinfo=timezone.utc
        )

        with self.assertRaisesRegex(ValueError, "current market session"):
            prediction_target_time(friday_at_two_et, 30, "5m")

    def test_intraday_target_uses_completed_bar_in_same_session(self):
        friday_at_one_et = datetime(
            2026, 8, 28, 17, 0, tzinfo=timezone.utc
        )
        target = prediction_target_time(friday_at_one_et, 30, "5m")

        self.assertEqual(
            datetime(2026, 8, 28, 19, 35, tzinfo=timezone.utc),
            target,
        )

    def test_intraday_prediction_is_not_due_before_target_bar_closes(self):
        input_at = datetime(2026, 8, 28, 18, 29, tzinfo=timezone.utc)
        target_at = prediction_target_time(input_at, 5, "1m")
        self.record(
            input_at=input_at,
            target_at=target_at,
            interval="1m",
            horizon=5,
        )

        result = self.store.settle_due(
            now=datetime(2026, 8, 28, 18, 33, 59, tzinfo=timezone.utc),
            price_fetcher=lambda ticker, target, interval: [(target, 102.0)],
        )

        self.assertEqual(
            datetime(2026, 8, 28, 18, 35, tzinfo=timezone.utc),
            target_at,
        )
        self.assertEqual(0, result["due"])

    def test_settlement_rejects_observations_far_after_target(self):
        self.record()

        result = self.store.settle_due(
            now=datetime(2026, 9, 10, tzinfo=timezone.utc),
            price_fetcher=lambda ticker, target, interval: [
                (target + timedelta(days=9), 102.0),
            ],
        )

        self.assertEqual(0, result["settled"])
        self.assertEqual(1, result["unavailable"])

    def test_legacy_schema_is_migrated_without_losing_predictions(self):
        database = Path(self.temporary_directory.name) / "legacy.db"
        with closing(sqlite3.connect(database)) as connection:
            connection.executescript(
                """
                CREATE TABLE models (
                    model_name TEXT PRIMARY KEY,
                    family TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    interval TEXT NOT NULL,
                    last_trained_at TEXT,
                    training_metrics_json TEXT NOT NULL DEFAULT '{}',
                    target_accuracy REAL NOT NULL DEFAULT 0.70,
                    minimum_accuracy REAL NOT NULL DEFAULT 0.65,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_at TEXT NOT NULL,
                    target_at TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    base_price REAL NOT NULL,
                    predicted_return_pct REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    actual_price REAL,
                    actual_at TEXT,
                    hit INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(model_name, ticker, input_at, target_at)
                );
                INSERT INTO models (
                    model_name, family, horizon, interval, updated_at
                ) VALUES (
                    'model_short_1d', 'HorizonModel', 1, '1d',
                    '2026-08-28T20:00:00Z'
                );
                INSERT INTO predictions (
                    model_name, ticker, created_at, input_at, target_at,
                    interval, horizon, base_price, predicted_return_pct,
                    predicted_price, direction, confidence
                ) VALUES (
                    'model_short_1d', 'AAPL', '2026-08-28T20:00:00Z',
                    '2026-08-28T20:00:00Z', '2026-08-31T20:00:00Z',
                    '1d', 1, 100, 2, 102, 'UP', 0.72
                );
                """
            )
            connection.commit()

        migrated = PredictionStore(database)
        history = migrated.get_history()
        new_version = migrated.record_prediction(
            model_name="model_short_1d",
            family="HorizonModel",
            ticker="AAPL",
            input_at=self.input_at,
            target_at=self.target_at,
            interval="1d",
            horizon=1,
            base_price=100,
            predicted_return_pct=3,
            direction="UP",
            confidence=0.8,
            last_trained_at="2026-08-29T20:00:00Z",
        )

        self.assertEqual("unknown", history[0]["model_trained_at"])
        self.assertNotEqual(history[0]["id"], new_version["id"])
        self.assertEqual(2, len(migrated.get_history()))


if __name__ == "__main__":
    unittest.main()
