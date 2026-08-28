"""Small persistent stale-while-refresh cache for web-facing market data."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)
DEFAULT_CACHE_PATH = (
    Path(__file__).parent.parent / "ml" / "models" / "web_response_cache.json"
)


class PersistentTTLCache:
    """Return stale data immediately while refreshing it in bounded workers."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_workers: int = 6,
        max_entries: int = 500,
    ):
        configured_path = os.getenv("WEBAPP_CACHE_PATH")
        self.path = Path(path or configured_path or DEFAULT_CACHE_PATH)
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._persist_lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._in_flight: dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="web-cache",
        )
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._entries = data
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError):
            logger.warning("Ignoring invalid web cache at %s", self.path)

    def _persist(self) -> None:
        with self._persist_lock:
            with self._lock:
                entries = dict(self._entries)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(
                f"{self.path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                temporary.write_text(
                    json.dumps(entries, separators=(",", ":")),
                    encoding="utf-8",
                )
                os.replace(temporary, self.path)
            except (OSError, TypeError, ValueError):
                logger.exception("Unable to persist web response cache")
                temporary.unlink(missing_ok=True)

    def _store_completed(self, key: str, future: Future) -> None:
        try:
            value = future.result()
        except Exception as exc:
            logger.warning("Background refresh failed for %s: %s", key, exc)
        else:
            now = time.time()
            with self._lock:
                self._entries[key] = {"updated_at": now, "value": value}
                if len(self._entries) > self.max_entries:
                    oldest = sorted(
                        self._entries,
                        key=lambda item: self._entries[item].get("updated_at", 0),
                    )
                    for expired_key in oldest[: len(self._entries) - self.max_entries]:
                        self._entries.pop(expired_key, None)
            self._persist()
        finally:
            with self._lock:
                if self._in_flight.get(key) is future:
                    self._in_flight.pop(key, None)

    def _refresh(self, key: str, loader: Callable[[], Any]) -> Future:
        with self._lock:
            future = self._in_flight.get(key)
            if future is None:
                future = self._executor.submit(loader)
                self._in_flight[key] = future
                future.add_done_callback(
                    lambda completed, cache_key=key: self._store_completed(
                        cache_key,
                        completed,
                    )
                )
            return future

    def get(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        ttl_seconds: float,
        wait_seconds: float = 3.5,
        fallback: Any = None,
    ) -> tuple[Any, str]:
        """Return ``(value, state)`` where state describes cache freshness."""
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if (
                entry is not None
                and now - float(entry.get("updated_at", 0)) <= ttl_seconds
            ):
                return entry["value"], "fresh"

        future = self._refresh(key, loader)
        if entry is not None:
            return entry["value"], "stale"

        try:
            return future.result(timeout=max(0.0, wait_seconds)), "loaded"
        except TimeoutError:
            return fallback, "refreshing"
        except Exception:
            return fallback, "error"

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate one key or the entire in-memory and persistent cache."""
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries.pop(key, None)
        self._persist()

    def close(self) -> None:
        """Release worker threads; primarily useful in tests."""
        self._executor.shutdown(wait=True, cancel_futures=True)
