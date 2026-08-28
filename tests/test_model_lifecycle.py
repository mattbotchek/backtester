import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.features import create_intraday_targets
from ml.model import StockPredictor, reconcile_prediction
from ml.multi_horizon import HORIZONS, HorizonModel
from ml.prediction_store import PredictionStore
from ml.sentiment.combined_model import CombinedPredictor


class FakeTicker:
    def __init__(self, data):
        self.data = data

    def history(self, **kwargs):
        return self.data


class FakeScaler:
    def transform(self, values):
        return np.asarray(values)


class FakeClassifier:
    def predict(self, values):
        return np.array([1])

    def predict_proba(self, values):
        return np.array([[0.2, 0.8]])


class FakeRegressor:
    def predict(self, values):
        return np.array([0.02])


class ModelLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Path(self.directory.name) / "predictions.db"
        self.environment = patch.dict(
            os.environ,
            {"PREDICTION_DB_PATH": str(self.database)},
        )
        self.environment.start()
        self.index = pd.DatetimeIndex(["2026-08-28T20:00:00Z"])
        self.market_data = pd.DataFrame({"Close": [100.0]}, index=self.index)
        self.features = pd.DataFrame({"signal": [1.0]}, index=self.index)
        self.trained_at = "2026-08-28T19:00:00Z"

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def configure(self, model):
        model.scaler = FakeScaler()
        model.classifier = FakeClassifier()
        model.regressor = FakeRegressor()
        model.feature_names = ["signal"]
        model.is_trained = True
        model._training_stats = {"trained_at": self.trained_at}
        return model

    def test_every_model_family_records_the_shared_lifecycle(self):
        default = self.configure(StockPredictor(horizon=1))
        horizon = self.configure(HorizonModel(HORIZONS["short_1d"]))
        combined = self.configure(
            CombinedPredictor(horizon=1, use_sentiment=False, use_macro=False)
        )

        with (
            patch("yfinance.Ticker", return_value=FakeTicker(self.market_data)),
            patch("ml.model.compute_features", return_value=self.features),
            patch("ml.multi_horizon.compute_features", return_value=self.features),
            patch(
                "ml.sentiment.combined_model.compute_features",
                return_value=self.features,
            ),
            patch("ml.sentiment.combined_model._download_context", return_value={}),
        ):
            default_prediction = default.predict("AAPL")
            horizon_prediction = horizon.predict_ticker("AAPL")
            combined_prediction = combined.predict("AAPL")

        records = PredictionStore(self.database).get_history(ticker="AAPL")
        self.assertEqual(3, len(records))
        self.assertEqual(
            {"StockPredictor", "HorizonModel", "CombinedPredictor"},
            {
                PredictionStore(self.database)
                .model_statistics(record["model_name"])[0]["family"]
                for record in records
            },
        )
        for record in records:
            self.assertEqual(self.trained_at, record["model_trained_at"])
            self.assertEqual("pending", record["status"])
            self.assertEqual(100.0, record["base_price"])
            self.assertEqual(102.0, record["predicted_price"])
            self.assertGreater(record["target_at"], record["input_at"])

        self.assertEqual(self.trained_at, default_prediction.last_trained_at)
        self.assertEqual(self.trained_at, horizon_prediction["last_trained_at"])
        self.assertEqual(self.trained_at, combined_prediction["last_trained_at"])

    def test_classifier_direction_controls_the_deployed_price_target(self):
        direction, predicted_return = reconcile_prediction(1, -0.02)
        self.assertEqual("UP", direction)
        self.assertEqual(0.02, predicted_return)

        direction, predicted_return = reconcile_prediction(-1, 0.02)
        self.assertEqual("DOWN", direction)
        self.assertEqual(-0.02, predicted_return)

    def test_intraday_training_target_does_not_cross_sessions(self):
        index = pd.DatetimeIndex([
            "2026-08-27T19:59:00Z",
            "2026-08-27T20:00:00Z",
            "2026-08-28T13:30:00Z",
            "2026-08-28T13:31:00Z",
        ])
        data = pd.DataFrame(
            {"Close": [100.0, 101.0, 200.0, 202.0]},
            index=index,
        )

        classification, returns = create_intraday_targets(
            data,
            horizon=1,
            threshold=0.0,
        )

        self.assertTrue(pd.isna(classification.iloc[1]))
        self.assertTrue(pd.isna(returns.iloc[1]))
        self.assertEqual(1, classification.iloc[0])
        self.assertAlmostEqual(0.01, returns.iloc[0])
