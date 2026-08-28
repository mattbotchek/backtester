import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyRegressor

from ml.compute import GPUXGBRegressor, io_worker_count, model_worker_count


class ComputeTests(unittest.TestCase):
    def test_cpu_reservation_does_not_reduce_io_parallelism(self):
        with (
            patch("ml.compute.cpu_count", return_value=16),
            patch.dict(
                os.environ,
                {
                    "ML_CPU_WORKERS": "0",
                    "ML_IO_WORKERS": "0",
                    "ML_RESERVED_CPUS": "2",
                },
            ),
        ):
            self.assertEqual(14, model_worker_count())
            self.assertEqual(16, io_worker_count())

    def test_cuda_fit_failure_retries_with_cpu_estimator(self):
        class FailingRegressor:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y):
                raise RuntimeError("CUDA unavailable")

        fake_xgboost = types.SimpleNamespace(XGBRegressor=FailingRegressor)
        model = GPUXGBRegressor(fallback=DummyRegressor(strategy="mean"))
        X = np.array([[0.0], [1.0]])
        y = np.array([2.0, 4.0])

        with patch.dict(sys.modules, {"xgboost": fake_xgboost}):
            model.fit(X, y)

        self.assertTrue(model.using_fallback_)
        np.testing.assert_allclose([3.0], model.predict([[2.0]]))


if __name__ == "__main__":
    unittest.main()
