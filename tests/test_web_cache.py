import tempfile
import threading
import time
import unittest
from pathlib import Path

from webapp.cache import PersistentTTLCache


class PersistentTTLCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "cache.json"
        self.cache = PersistentTTLCache(self.path, max_workers=2)

    def tearDown(self):
        self.cache.close()
        self.directory.cleanup()

    def test_fresh_values_avoid_reloading(self):
        calls = 0

        def load():
            nonlocal calls
            calls += 1
            return {"value": calls}

        first, first_state = self.cache.get("key", load, ttl_seconds=60)
        second, second_state = self.cache.get("key", load, ttl_seconds=60)

        self.assertEqual({"value": 1}, first)
        self.assertEqual({"value": 1}, second)
        self.assertEqual("loaded", first_state)
        self.assertEqual("fresh", second_state)
        self.assertEqual(1, calls)

    def test_stale_values_return_while_one_refresh_runs(self):
        first, _ = self.cache.get(
            "key",
            lambda: {"value": 1},
            ttl_seconds=60,
        )
        self.cache._entries["key"]["updated_at"] = 0
        release = threading.Event()

        def refresh():
            release.wait(timeout=2)
            return {"value": 2}

        started = time.perf_counter()
        stale, state = self.cache.get(
            "key",
            refresh,
            ttl_seconds=60,
        )
        elapsed = time.perf_counter() - started
        release.set()

        self.assertEqual({"value": 1}, first)
        self.assertEqual({"value": 1}, stale)
        self.assertEqual("stale", state)
        self.assertLess(elapsed, 0.2)

    def test_cold_timeout_returns_fallback_within_budget(self):
        release = threading.Event()

        def slow_load():
            release.wait(timeout=2)
            return {"value": 1}

        started = time.perf_counter()
        value, state = self.cache.get(
            "key",
            slow_load,
            ttl_seconds=60,
            wait_seconds=0.05,
            fallback={"refreshing": True},
        )
        elapsed = time.perf_counter() - started
        release.set()

        self.assertEqual({"refreshing": True}, value)
        self.assertEqual("refreshing", state)
        self.assertLess(elapsed, 0.5)
