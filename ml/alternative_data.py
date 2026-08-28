"""Point-in-time-safe collection and storage of public alternative data.

The collector intentionally uses only unauthenticated public endpoints.  Data is
stored with three distinct timestamps:

``event_time``
    When the underlying event happened (or the snapshot was taken).
``available_at``
    The earliest time at which this particular observation was public.
``collected_at``
    When this process observed it.

Feature queries enforce both ``event_time <= as_of`` and
``available_at <= as_of``.  The latter is the important anti-leakage barrier.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import feedparser
import pandas as pd
import requests
import yfinance

from .compute import parallel_map

logger = logging.getLogger(__name__)
UTC = timezone.utc

DEFAULT_DATABASE_PATH = Path(__file__).parent / "models" / "alternative_data.db"
DEFAULT_NEWS_RSS_FEEDS = {
    "cnbc_top": (
        "https://search.cnbc.com/rs/search/combinedcms/view.xml"
        "?partnerId=wrss01&id=100003114"
    ),
    "marketwatch_top": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "reuters_business": (
        "https://news.google.com/rss/search"
        "?q=site%3Areuters.com%20business%20finance"
        "&hl=en-US&gl=US&ceid=US%3Aen"
    ),
}
USGS_URL = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
    "all_week.geojson"
)
EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events?status=open&limit=200"
GDELT_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query=%28conflict%20OR%20war%20OR%20sanctions%20OR%20coup%29"
    "&mode=ArtList&maxrecords=75&format=json&timespan=1d"
)

FEATURE_NAMES = (
    "news_count",
    "news_sentiment_mean",
    "sec_filing_count",
    "sec_form4_count",
    "insider_trade_count",
    "insider_buy_count",
    "insider_sell_count",
    "insider_net_value",
    "congressional_trade_count",
    "congressional_buy_count",
    "congressional_sell_count",
    "congressional_net_value",
    "executive_post_count",
    "executive_post_sentiment_mean",
    "disaster_event_count",
    "disaster_severity_sum",
    "geopolitical_event_count",
    "geopolitical_sentiment_mean",
    "fundamentals_snapshot_count",
    "market_cap_latest",
    "pe_ratio_latest",
    "price_to_book_latest",
    "revenue_growth_latest",
    "earnings_event_count",
    "earnings_surprise_mean",
    "options_snapshot_count",
    "options_put_call_volume_ratio_latest",
    "options_implied_volatility_mean_latest",
    "custom_event_count",
    "custom_value_sum",
)

_NEGATIVE_WORDS = {
    "attack", "bankruptcy", "conflict", "crash", "crisis", "coup", "decline",
    "disaster", "earthquake", "fraud", "invasion", "loss", "missile", "plunge",
    "recession", "sanction", "selloff", "war",
}
_POSITIVE_WORDS = {
    "beat", "ceasefire", "gain", "growth", "peace", "profit", "rally",
    "recovery", "surge", "upgrade",
}


class AlternativeDataError(RuntimeError):
    """Base error for alternative-data ingestion."""


class ConfigurationError(AlternativeDataError):
    """Raised when a source is enabled without required configuration."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: Any, *, fallback: datetime | None = None,
            end_of_day: bool = False) -> datetime:
    """Parse common API timestamps into an aware UTC datetime."""
    if value is None or value == "":
        if fallback is None:
            raise ValueError("timestamp is required")
        return fallback.astimezone(UTC)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(
            value,
            datetime_time.max if end_of_day else datetime_time.min,
            UTC,
        )
    elif isinstance(value, (int, float)):
        numeric = float(value)
        if abs(numeric) > 10_000_000_000:
            numeric /= 1000.0
        parsed = datetime.fromtimestamp(numeric, UTC)
    else:
        text = str(value).strip()
        if not text:
            if fallback is None:
                raise ValueError("timestamp is empty")
            return fallback.astimezone(UTC)
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            normalized = text.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                parsed_date = date.fromisoformat(text[:10])
                parsed = datetime.combine(
                    parsed_date,
                    datetime_time.max if end_of_day else datetime_time.min,
                    UTC,
                )
        if len(text) == 10 and end_of_day:
            parsed = datetime.combine(parsed.date(), datetime_time.max, UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(value: Any, *, fallback: datetime | None = None,
         end_of_day: bool = False) -> str:
    return _as_utc(
        value, fallback=fallback, end_of_day=end_of_day
    ).isoformat(timespec="microseconds")


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return default


def _sentiment(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    positive = sum(word in _POSITIVE_WORDS for word in words)
    negative = sum(word in _NEGATIVE_WORDS for word in words)
    return max(-1.0, min(1.0, (positive - negative) / max(positive + negative, 1)))


def _endpoint_config(
    environment_name: str,
    defaults: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    raw = os.getenv(environment_name)
    if not raw:
        return list((defaults or {}).items())
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(parsed, Mapping):
        return [(str(name), str(url)) for name, url in parsed.items()]
    if not isinstance(parsed, list):
        raise ConfigurationError(f"{environment_name} must be JSON or a URL list")
    endpoints: list[tuple[str, str]] = []
    for index, item in enumerate(parsed):
        if isinstance(item, str):
            endpoints.append((f"{environment_name.lower()}_{index + 1}", item))
        elif isinstance(item, Mapping) and item.get("url"):
            endpoints.append(
                (str(item.get("name") or f"feed_{index + 1}"), str(item["url"]))
            )
        else:
            raise ConfigurationError(
                f"{environment_name} item {index} must be a URL or name/url object"
            )
    return endpoints


@dataclass(frozen=True)
class AlternativeDataEvent:
    """Normalized observation accepted by :class:`AlternativeDataStore`."""

    category: str
    source: str
    event_time: datetime | str
    available_at: datetime | str
    collected_at: datetime | str
    ticker: str | None = None
    external_id: str | None = None
    value: float | None = None
    sentiment: float | None = None
    payload: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SourceStatus:
    source: str
    status: str
    started_at: str
    completed_at: str
    fetched: int
    inserted: int
    error_type: str | None = None
    error: str | None = None


class AlternativeDataStore:
    """Durable, idempotent SQLite event store with point-in-time queries."""

    def __init__(self, database_path: str | Path | None = None):
        configured = database_path or os.getenv("ALT_DATA_DB_PATH")
        self.database_path = Path(configured or DEFAULT_DATABASE_PATH)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS alternative_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    ticker TEXT,
                    external_id TEXT,
                    event_time TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    collected_at TEXT NOT NULL,
                    value REAL,
                    sentiment REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_alt_events_point_in_time
                    ON alternative_events(ticker, available_at, event_time);
                CREATE INDEX IF NOT EXISTS idx_alt_events_category
                    ON alternative_events(category, available_at);

                CREATE TABLE IF NOT EXISTS alternative_source_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    fetched INTEGER NOT NULL,
                    inserted INTEGER NOT NULL,
                    error_type TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_alt_source_runs
                    ON alternative_source_runs(source, completed_at);
                """
            )

    @staticmethod
    def _normalized_event(event: AlternativeDataEvent | Mapping[str, Any]) -> dict:
        values = asdict(event) if isinstance(event, AlternativeDataEvent) else dict(event)
        category = str(values.get("category") or "").strip()
        source = str(values.get("source") or "").strip()
        if not category or not source:
            raise ValueError("event category and source are required")
        collected = _as_utc(values.get("collected_at"), fallback=_utc_now())
        event_time = _as_utc(values.get("event_time"), fallback=collected)
        available = _as_utc(values.get("available_at"), fallback=collected)
        ticker = str(values.get("ticker") or "").strip().upper() or None
        payload = values.get("payload") or {}
        if not isinstance(payload, Mapping):
            raise ValueError("event payload must be a mapping")
        payload_json = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        external_id = values.get("external_id")
        identity = {
            "category": category,
            "source": source,
            "ticker": ticker,
            "external_id": str(external_id) if external_id is not None else None,
            "event_time": event_time.isoformat(timespec="microseconds"),
            "payload": payload if external_id is None else None,
        }
        dedupe_key = hashlib.sha256(
            json.dumps(
                identity, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()
        value = values.get("value")
        sentiment = values.get("sentiment")
        return {
            "dedupe_key": dedupe_key,
            "category": category,
            "source": source,
            "ticker": ticker,
            "external_id": str(external_id) if external_id is not None else None,
            "event_time": event_time.isoformat(timespec="microseconds"),
            "available_at": available.isoformat(timespec="microseconds"),
            "collected_at": collected.isoformat(timespec="microseconds"),
            "value": None if value is None else _finite_number(value),
            "sentiment": None if sentiment is None else _finite_number(sentiment),
            "payload_json": payload_json,
        }

    def add_event(self, event: AlternativeDataEvent | Mapping[str, Any]) -> bool:
        """Insert one event, returning ``False`` when it is a duplicate."""
        return self.add_events([event]) == 1

    def add_events(
        self, events: Iterable[AlternativeDataEvent | Mapping[str, Any]]
    ) -> int:
        """Atomically insert normalized events with deterministic deduplication."""
        normalized = [self._normalized_event(event) for event in events]
        if not normalized:
            return 0
        with self._connection() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO alternative_events (
                    dedupe_key, category, source, ticker, external_id,
                    event_time, available_at, collected_at, value, sentiment,
                    payload_json
                ) VALUES (
                    :dedupe_key, :category, :source, :ticker, :external_id,
                    :event_time, :available_at, :collected_at, :value,
                    :sentiment, :payload_json
                )
                """,
                normalized,
            )
            return connection.total_changes - before

    def record_source_status(self, status: SourceStatus | Mapping[str, Any]) -> None:
        values = asdict(status) if isinstance(status, SourceStatus) else dict(status)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO alternative_source_runs (
                    source, status, started_at, completed_at, fetched, inserted,
                    error_type, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    values["source"], values["status"], values["started_at"],
                    values["completed_at"], int(values.get("fetched", 0)),
                    int(values.get("inserted", 0)), values.get("error_type"),
                    values.get("error"),
                ),
            )

    def count_events(self) -> int:
        with self._connection() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM alternative_events"
            ).fetchone()[0])

    def _point_in_time_rows(
        self, ticker: str, as_of: datetime | str, lookback_days: int
    ) -> list[sqlite3.Row]:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        as_of_dt = _as_utc(as_of)
        cutoff = as_of_dt - timedelta(days=lookback_days)
        with self._connection() as connection:
            return connection.execute(
                """
                SELECT * FROM alternative_events
                WHERE event_time >= ?
                  AND event_time <= ?
                  AND available_at <= ?
                  AND (
                      ticker IS NULL OR ticker = '' OR ticker = 'GLOBAL'
                      OR ticker = ?
                  )
                ORDER BY available_at, id
                """,
                (
                    cutoff.isoformat(timespec="microseconds"),
                    as_of_dt.isoformat(timespec="microseconds"),
                    as_of_dt.isoformat(timespec="microseconds"),
                    ticker.upper(),
                ),
            ).fetchall()

    @staticmethod
    def _features_from_rows(rows: Sequence[sqlite3.Row]) -> dict[str, float]:
        features = {name: 0.0 for name in FEATURE_NAMES}
        sentiments: dict[str, list[float]] = {
            "news": [],
            "executive_post": [],
            "geopolitical": [],
            "earnings": [],
        }
        latest: dict[str, tuple[str, Mapping[str, Any]]] = {}

        for row in rows:
            category = row["category"]
            payload = json.loads(row["payload_json"])
            if category == "news":
                features["news_count"] += 1
                sentiments["news"].append(_finite_number(row["sentiment"]))
            elif category == "sec_filing":
                features["sec_filing_count"] += 1
                if str(payload.get("form", "")).upper().replace(" ", "") in {
                    "4", "4/A",
                }:
                    features["sec_form4_count"] += 1
            elif category == "insider_trade":
                features["insider_trade_count"] += 1
                transaction = str(payload.get("transaction_type", "")).upper()
                trade_value = abs(_finite_number(row["value"]))
                if transaction in {"P", "BUY", "PURCHASE"}:
                    features["insider_buy_count"] += 1
                    features["insider_net_value"] += trade_value
                elif transaction in {"S", "SELL", "SALE"}:
                    features["insider_sell_count"] += 1
                    features["insider_net_value"] -= trade_value
            elif category == "congressional_trade":
                features["congressional_trade_count"] += 1
                transaction = str(payload.get("transaction_type", "")).lower()
                trade_value = abs(_finite_number(row["value"]))
                if "purchase" in transaction or transaction == "buy":
                    features["congressional_buy_count"] += 1
                    features["congressional_net_value"] += trade_value
                elif "sale" in transaction or transaction == "sell":
                    features["congressional_sell_count"] += 1
                    features["congressional_net_value"] -= trade_value
            elif category == "executive_post":
                features["executive_post_count"] += 1
                sentiments["executive_post"].append(
                    _finite_number(row["sentiment"])
                )
            elif category == "disaster":
                features["disaster_event_count"] += 1
                features["disaster_severity_sum"] += _finite_number(row["value"])
            elif category == "geopolitical":
                features["geopolitical_event_count"] += 1
                sentiments["geopolitical"].append(
                    _finite_number(row["sentiment"])
                )
            elif category == "fundamentals":
                features["fundamentals_snapshot_count"] += 1
                latest["fundamentals"] = (row["available_at"], payload)
            elif category == "earnings":
                features["earnings_event_count"] += 1
                sentiments["earnings"].append(_finite_number(row["value"]))
            elif category == "options":
                features["options_snapshot_count"] += 1
                latest["options"] = (row["available_at"], payload)
            elif category == "custom_event" or category.startswith("custom:"):
                features["custom_event_count"] += 1
                features["custom_value_sum"] += _finite_number(row["value"])

        for category, feature in (
            ("news", "news_sentiment_mean"),
            ("executive_post", "executive_post_sentiment_mean"),
            ("geopolitical", "geopolitical_sentiment_mean"),
            ("earnings", "earnings_surprise_mean"),
        ):
            values = sentiments[category]
            features[feature] = sum(values) / len(values) if values else 0.0

        fundamentals = latest.get("fundamentals", ("", {}))[1]
        features["market_cap_latest"] = _finite_number(
            fundamentals.get("market_cap")
        )
        features["pe_ratio_latest"] = _finite_number(
            fundamentals.get("pe_ratio")
        )
        features["price_to_book_latest"] = _finite_number(
            fundamentals.get("price_to_book")
        )
        features["revenue_growth_latest"] = _finite_number(
            fundamentals.get("revenue_growth")
        )
        options = latest.get("options", ("", {}))[1]
        features["options_put_call_volume_ratio_latest"] = _finite_number(
            options.get("put_call_volume_ratio")
        )
        features["options_implied_volatility_mean_latest"] = _finite_number(
            options.get("implied_volatility_mean")
        )
        return features

    def get_feature_frame(
        self,
        ticker: str,
        as_of_values: Sequence[Any] | pd.Index,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """Build a fixed-schema feature frame with one batched database read."""
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")

        original_index = pd.Index(as_of_values)
        if original_index.empty:
            return pd.DataFrame(
                index=original_index,
                columns=[f"alt_{name}" for name in FEATURE_NAMES],
                dtype=float,
            )

        as_of_datetimes = [_as_utc(value) for value in original_index]
        earliest = min(as_of_datetimes) - timedelta(days=lookback_days)
        latest = max(as_of_datetimes)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM alternative_events
                WHERE event_time >= ?
                  AND event_time <= ?
                  AND available_at <= ?
                  AND (
                      ticker IS NULL OR ticker = '' OR ticker = 'GLOBAL'
                      OR ticker = ?
                  )
                ORDER BY available_at, id
                """,
                (
                    earliest.isoformat(timespec="microseconds"),
                    latest.isoformat(timespec="microseconds"),
                    latest.isoformat(timespec="microseconds"),
                    ticker.upper(),
                ),
            ).fetchall()

        timestamped_rows = [
            (
                row,
                _as_utc(row["event_time"]),
                _as_utc(row["available_at"]),
            )
            for row in rows
        ]
        records = []
        for as_of_dt in as_of_datetimes:
            cutoff = as_of_dt - timedelta(days=lookback_days)
            eligible = [
                row
                for row, event_time, available_at in timestamped_rows
                if cutoff <= event_time <= as_of_dt
                and available_at <= as_of_dt
            ]
            records.append({
                f"alt_{name}": value
                for name, value in self._features_from_rows(eligible).items()
            })
        return pd.DataFrame(records, index=original_index, dtype=float)

    def get_point_in_time_features(
        self,
        ticker: str,
        as_of: datetime | str,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """Return fixed-schema numeric features and separate provenance metadata."""
        as_of_dt = _as_utc(as_of)
        rows = self._point_in_time_rows(ticker, as_of_dt, lookback_days)
        features = self._features_from_rows(rows)
        source_metadata: dict[str, dict[str, Any]] = {}
        category_counts: dict[str, int] = {}

        for row in rows:
            category = row["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
            source = source_metadata.setdefault(
                row["source"], {"row_count": 0, "latest_available_at": None}
            )
            source["row_count"] += 1
            if (
                source["latest_available_at"] is None
                or row["available_at"] > source["latest_available_at"]
            ):
                source["latest_available_at"] = row["available_at"]

        for details in source_metadata.values():
            latest_available = _as_utc(details["latest_available_at"])
            details["freshness_hours"] = max(
                0.0, (as_of_dt - latest_available).total_seconds() / 3600.0
            )

        with self._connection() as connection:
            latest_runs = connection.execute(
                """
                SELECT r.* FROM alternative_source_runs r
                JOIN (
                    SELECT source, MAX(id) AS latest_id
                    FROM alternative_source_runs GROUP BY source
                ) x ON x.latest_id = r.id
                ORDER BY r.source
                """
            ).fetchall()
        run_metadata = {
            row["source"]: {
                "status": row["status"],
                "completed_at": row["completed_at"],
                "fetched": row["fetched"],
                "inserted": row["inserted"],
                "error_type": row["error_type"],
                "error": row["error"],
            }
            for row in latest_runs
        }
        return {
            "features": features,
            "metadata": {
                "ticker": ticker.upper(),
                "as_of": as_of_dt.isoformat(timespec="microseconds"),
                "lookback_days": lookback_days,
                "included_rows": len(rows),
                "category_counts": category_counts,
                "sources": source_metadata,
                "latest_collection_status": run_metadata,
            },
        }


class _RateLimiter:
    def __init__(
        self, minimum_interval: float, sleep: Callable[[float], None] = time.sleep
    ):
        self.minimum_interval = max(0.0, minimum_interval)
        self.sleep = sleep
        self._last_request: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_request is not None:
                remaining = self.minimum_interval - (now - self._last_request)
                if remaining > 0:
                    self.sleep(remaining)
            self._last_request = time.monotonic()


class AlternativeDataCollector:
    """Collect public feeds into an :class:`AlternativeDataStore`.

    Endpoint lists may be supplied explicitly for tests/applications or loaded
    from the documented ``ALT_DATA_*`` environment variables.
    """

    def __init__(
        self,
        store: AlternativeDataStore | None = None,
        *,
        session: requests.Session | Any | None = None,
        timeout_seconds: float | None = None,
        now: Callable[[], datetime] = _utc_now,
        yfinance_factory: Callable[[str], Any] = yfinance.Ticker,
        news_feeds: Sequence[tuple[str, str]] | None = None,
        executive_feeds: Sequence[tuple[str, str]] | None = None,
        congressional_endpoints: Sequence[tuple[str, str]] | None = None,
        custom_json_feeds: Sequence[tuple[str, str]] | None = None,
        enable_usgs: bool | None = None,
        enable_eonet: bool | None = None,
        enable_gdelt: bool | None = None,
    ):
        self.store = store or AlternativeDataStore()
        self.session = session or requests.Session()
        self.timeout_seconds = float(
            timeout_seconds or os.getenv("ALT_DATA_HTTP_TIMEOUT_SECONDS", "15")
        )
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.now = now
        self.yfinance_factory = yfinance_factory
        self.sec_user_agent = os.getenv("ALT_DATA_SEC_USER_AGENT", "").strip()
        sec_interval = float(os.getenv("ALT_DATA_SEC_RATE_LIMIT_SECONDS", "0.12"))
        self._sec_limiter = _RateLimiter(sec_interval)
        self.news_feeds = list(news_feeds) if news_feeds is not None else (
            _endpoint_config("ALT_DATA_NEWS_RSS_FEEDS", DEFAULT_NEWS_RSS_FEEDS)
        )
        self.executive_feeds = (
            list(executive_feeds) if executive_feeds is not None
            else _endpoint_config("ALT_DATA_EXECUTIVE_RSS_FEEDS")
        )
        self.congressional_endpoints = (
            list(congressional_endpoints)
            if congressional_endpoints is not None
            else _endpoint_config("ALT_DATA_CONGRESSIONAL_ENDPOINTS")
        )
        self.custom_json_feeds = (
            list(custom_json_feeds) if custom_json_feeds is not None
            else _endpoint_config("ALT_DATA_CUSTOM_JSON_FEEDS")
        )
        self.enable_usgs = self._env_bool(
            "ALT_DATA_ENABLE_USGS", True, enable_usgs
        )
        self.enable_eonet = self._env_bool(
            "ALT_DATA_ENABLE_EONET", True, enable_eonet
        )
        self.enable_gdelt = self._env_bool(
            "ALT_DATA_ENABLE_GDELT", True, enable_gdelt
        )
        self._active_tickers: set[str] = set()
        self._cik_by_ticker: dict[str, int] | None = None
        self._cik_lock = threading.Lock()

    @staticmethod
    def _env_bool(name: str, default: bool, explicit: bool | None) -> bool:
        if explicit is not None:
            return explicit
        value = os.getenv(name)
        return default if value is None else value.lower() not in {
            "0", "false", "no", "off"
        }

    def _get_json(
        self, url: str, *, headers: Mapping[str, str] | None = None,
        sec: bool = False
    ) -> Any:
        if sec:
            self._sec_limiter.wait()
        response = self.session.get(
            url, headers=dict(headers or {}), timeout=self.timeout_seconds
        )
        response.raise_for_status()
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError) as error:
            raise AlternativeDataError(f"invalid JSON from {url}: {error}") from error

    def _get_content(
        self, url: str, *, headers: Mapping[str, str] | None = None,
        sec: bool = False
    ) -> bytes:
        if sec:
            self._sec_limiter.wait()
        response = self.session.get(
            url, headers=dict(headers or {}), timeout=self.timeout_seconds
        )
        response.raise_for_status()
        return response.content

    def _run_source(
        self, source: str, collect: Callable[[], tuple[int, int]]
    ) -> SourceStatus:
        started = self.now().astimezone(UTC)
        try:
            fetched, inserted = collect()
            status = SourceStatus(
                source, "ok", _iso(started), _iso(self.now()), fetched, inserted
            )
        except Exception as error:
            logger.warning("Alternative-data source %s failed: %s", source, error)
            status = SourceStatus(
                source=source,
                status="error",
                started_at=_iso(started),
                completed_at=_iso(self.now()),
                fetched=0,
                inserted=0,
                error_type=type(error).__name__,
                error=str(error),
            )
        self.store.record_source_status(status)
        return status

    def collect_all(
        self,
        tickers: Sequence[str],
        sources: set[str] | None = None,
    ) -> dict[str, Any]:
        """Collect enabled sources while isolating and reporting every failure."""
        selected = sources or {
            "news", "sec", "congressional", "executive", "events",
            "yfinance", "custom",
        }
        self._active_tickers = {
            ticker.strip().upper() for ticker in tickers if ticker.strip()
        }
        jobs: list[tuple[str, Callable[[], tuple[int, int]]]] = []

        if "news" in selected:
            for name, url in self.news_feeds:
                jobs.append((
                    f"news:{name}",
                    lambda name=name, url=url: self.collect_rss_feed(
                        name, url, category="news"
                    ),
                ))
        if "executive" in selected:
            for name, url in self.executive_feeds:
                jobs.append((
                    f"executive:{name}",
                    lambda name=name, url=url: self.collect_rss_feed(
                        name, url, category="executive_post"
                    ),
                ))
        if "congressional" in selected:
            for name, url in self.congressional_endpoints:
                jobs.append((
                    f"congressional:{name}",
                    lambda name=name, url=url: self.collect_congressional(
                        name, url
                    ),
                ))
        if "custom" in selected:
            for name, url in self.custom_json_feeds:
                jobs.append((
                    f"custom:{name}",
                    lambda name=name, url=url: self.collect_custom_json(name, url),
                ))
        if "events" in selected:
            if self.enable_usgs:
                jobs.append(("usgs", self.collect_usgs))
            if self.enable_eonet:
                jobs.append(("nasa_eonet", self.collect_eonet))
            if self.enable_gdelt:
                jobs.append(("gdelt", self.collect_gdelt))
        if "sec" in selected:
            for ticker in sorted(self._active_tickers):
                jobs.append((
                    f"sec:{ticker}",
                    lambda ticker=ticker: self.collect_sec(ticker),
                ))
        if "yfinance" in selected:
            for ticker in sorted(self._active_tickers):
                jobs.append((
                    f"yfinance:{ticker}",
                    lambda ticker=ticker: self.collect_yfinance(ticker),
                ))

        statuses = parallel_map(
            lambda job: self._run_source(job[0], job[1]),
            jobs,
        )

        return {
            "statuses": [asdict(status) for status in statuses],
            "fetched": sum(status.fetched for status in statuses),
            "inserted": sum(status.inserted for status in statuses),
            "ok": sum(status.status == "ok" for status in statuses),
            "failed": sum(status.status == "error" for status in statuses),
        }

    def _ticker_from_text(self, text: str) -> str | None:
        for ticker in sorted(self._active_tickers, key=len, reverse=True):
            if re.search(rf"(?<![A-Z])\$?{re.escape(ticker)}(?![A-Z])", text.upper()):
                return ticker
        return None

    def collect_rss_feed(
        self, name: str, url: str, *, category: str
    ) -> tuple[int, int]:
        content = self._get_content(url)
        parsed = feedparser.parse(content)
        if getattr(parsed, "bozo", False) and not parsed.entries:
            raise AlternativeDataError(
                f"invalid RSS/Atom from {url}: {parsed.get('bozo_exception')}"
            )
        collected = self.now().astimezone(UTC)
        events = []
        for index, entry in enumerate(parsed.entries):
            title = str(entry.get("title", ""))
            summary = str(entry.get("summary", entry.get("description", "")))
            published = _first(
                entry, "published", "updated", "created", default=None
            )
            event_time = _as_utc(published, fallback=collected)
            link = str(entry.get("link", ""))
            identifier = str(entry.get("id") or link or f"{index}:{title}")
            text = f"{title} {summary}"
            events.append(AlternativeDataEvent(
                category=category,
                source=name,
                ticker=self._ticker_from_text(text),
                external_id=identifier,
                event_time=event_time,
                available_at=event_time if published else collected,
                collected_at=collected,
                sentiment=_sentiment(text),
                payload={"title": title, "summary": summary, "url": link},
            ))
        return len(events), self.store.add_events(events)

    @staticmethod
    def _records_from_json(data: Any) -> list[Mapping[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, Mapping)]
        if not isinstance(data, Mapping):
            raise AlternativeDataError("feed JSON must be an object or array")
        for key in ("events", "trades", "results", "data", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, Mapping)]
            if isinstance(value, Mapping):
                nested = AlternativeDataCollector._records_from_json(value)
                if nested:
                    return nested
        return [data] if data else []

    def collect_congressional(self, name: str, url: str) -> tuple[int, int]:
        records = self._records_from_json(self._get_json(url))
        collected = self.now().astimezone(UTC)
        events = []
        for index, record in enumerate(records):
            ticker = str(_first(
                record, "ticker", "symbol", "assetTicker", default=""
            )).strip().upper() or None
            transaction_date = _first(
                record, "transaction_date", "transactionDate", "date",
                "transactionDateText",
            )
            disclosed = _first(
                record, "disclosure_date", "disclosureDate", "published_at",
                "filedDate", "reported_at",
            )
            event_time = _as_utc(transaction_date, fallback=collected)
            # Date-only disclosures are conservatively available at end-of-day.
            available = _as_utc(
                disclosed, fallback=collected,
                end_of_day=isinstance(disclosed, str) and len(disclosed) == 10,
            )
            transaction_type = str(_first(
                record, "transaction_type", "transactionType", "type",
                "transaction", default=""
            ))
            raw_amount = _first(
                record, "amount", "value", "amountTo", "amount_to", default=0
            )
            if isinstance(raw_amount, str):
                numbers = re.findall(r"[\d,.]+", raw_amount)
                amount = sum(
                    _finite_number(number.replace(",", "")) for number in numbers
                ) / max(len(numbers), 1)
            else:
                amount = _finite_number(raw_amount)
            external_id = str(_first(
                record, "id", "transaction_id", "disclosureId",
                default=f"{ticker}:{transaction_date}:{transaction_type}:{index}",
            ))
            events.append(AlternativeDataEvent(
                category="congressional_trade",
                source=name,
                ticker=ticker,
                external_id=external_id,
                event_time=event_time,
                available_at=available,
                collected_at=collected,
                value=amount,
                payload={
                    "transaction_type": transaction_type,
                    "representative": _first(
                        record, "representative", "politician", "member", "name"
                    ),
                    "raw": record,
                },
            ))
        return len(events), self.store.add_events(events)

    def collect_custom_json(self, name: str, url: str) -> tuple[int, int]:
        records = self._records_from_json(self._get_json(url))
        collected = self.now().astimezone(UTC)
        events = []
        for index, record in enumerate(records):
            if not record.get("event_time"):
                raise AlternativeDataError(
                    f"custom feed {name} record {index} lacks event_time"
                )
            available_raw = record.get("available_at")
            available = _as_utc(available_raw, fallback=collected)
            category = str(record.get("category") or "custom_event")
            if category not in {
                "news", "sec_filing", "insider_trade", "congressional_trade",
                "executive_post", "disaster", "geopolitical", "fundamentals",
                "earnings", "options", "custom_event",
            } and not category.startswith("custom:"):
                category = f"custom:{category}"
            payload = record.get("payload", record.get("attributes", {}))
            if not isinstance(payload, Mapping):
                raise AlternativeDataError(
                    f"custom feed {name} record {index} payload is not an object"
                )
            events.append(AlternativeDataEvent(
                category=category,
                source=str(record.get("source") or name),
                ticker=record.get("ticker"),
                external_id=str(record.get("id") or f"{index}"),
                event_time=record["event_time"],
                available_at=available,
                collected_at=collected,
                value=record.get("value"),
                sentiment=record.get("sentiment"),
                payload=payload,
            ))
        return len(events), self.store.add_events(events)

    def collect_usgs(self) -> tuple[int, int]:
        data = self._get_json(USGS_URL)
        collected = self.now().astimezone(UTC)
        events = []
        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            magnitude = _finite_number(properties.get("mag"))
            event_time = _as_utc(properties.get("time"), fallback=collected)
            available = _as_utc(properties.get("updated"), fallback=collected)
            events.append(AlternativeDataEvent(
                category="disaster",
                source="usgs",
                external_id=str(feature.get("id") or properties.get("code")),
                event_time=event_time,
                available_at=available,
                collected_at=collected,
                value=magnitude,
                payload={
                    "kind": "earthquake",
                    "magnitude": magnitude,
                    "place": properties.get("place"),
                    "url": properties.get("url"),
                    "geometry": feature.get("geometry"),
                },
            ))
        return len(events), self.store.add_events(events)

    def collect_eonet(self) -> tuple[int, int]:
        data = self._get_json(EONET_URL)
        collected = self.now().astimezone(UTC)
        events = []
        for event in data.get("events", []):
            geometry = event.get("geometry") or []
            event_time = _as_utc(
                geometry[-1].get("date") if geometry else None,
                fallback=collected,
            )
            categories = event.get("categories") or []
            events.append(AlternativeDataEvent(
                category="disaster",
                source="nasa_eonet",
                external_id=str(event.get("id")),
                event_time=event_time,
                # EONET does not expose first-publication time.
                available_at=collected,
                collected_at=collected,
                value=1.0,
                payload={
                    "title": event.get("title"),
                    "categories": categories,
                    "geometry": geometry[-1] if geometry else None,
                    "link": event.get("link"),
                },
            ))
        return len(events), self.store.add_events(events)

    def collect_gdelt(self) -> tuple[int, int]:
        data = self._get_json(GDELT_URL)
        collected = self.now().astimezone(UTC)
        events = []
        for index, article in enumerate(data.get("articles", [])):
            title = str(article.get("title", ""))
            seen = _first(article, "seendate", "date", default=None)
            event_time = _as_utc(seen, fallback=collected)
            url = str(article.get("url", ""))
            events.append(AlternativeDataEvent(
                category="geopolitical",
                source="gdelt",
                ticker=self._ticker_from_text(title),
                external_id=url or f"{index}:{title}",
                event_time=event_time,
                available_at=event_time if seen else collected,
                collected_at=collected,
                sentiment=_sentiment(title),
                payload={
                    "title": title,
                    "url": url,
                    "domain": article.get("domain"),
                    "language": article.get("language"),
                    "source_country": article.get("sourcecountry"),
                },
            ))
        return len(events), self.store.add_events(events)

    def _sec_headers(self) -> dict[str, str]:
        if not self.sec_user_agent:
            raise ConfigurationError(
                "ALT_DATA_SEC_USER_AGENT is required for SEC collection; "
                "set it to an application name and real contact email"
            )
        return {
            "User-Agent": self.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

    def _load_sec_tickers(self) -> dict[str, int]:
        if self._cik_by_ticker is not None:
            return self._cik_by_ticker
        with self._cik_lock:
            if self._cik_by_ticker is not None:
                return self._cik_by_ticker
            data = self._get_json(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": self.sec_user_agent},
                sec=True,
            )
            self._cik_by_ticker = {
                str(item["ticker"]).upper(): int(item["cik_str"])
                for item in data.values()
            }
        return self._cik_by_ticker

    @staticmethod
    def _sec_recent_rows(recent: Mapping[str, Sequence[Any]]) -> list[dict]:
        lengths = [len(values) for values in recent.values() if isinstance(values, list)]
        if not lengths:
            return []
        return [
            {
                key: values[index]
                for key, values in recent.items()
                if isinstance(values, list) and index < len(values)
            }
            for index in range(max(lengths))
        ]

    @staticmethod
    def _xml_descendant_text(node: ET.Element, local_name: str) -> str | None:
        for descendant in node.iter():
            if descendant.tag.rsplit("}", 1)[-1] == local_name:
                text = (descendant.text or "").strip()
                if text:
                    return text
                for value_node in descendant.iter():
                    if value_node.tag.rsplit("}", 1)[-1] == "value":
                        value = (value_node.text or "").strip()
                        if value:
                            return value
        return None

    def _form4_events(
        self,
        ticker: str,
        cik: int,
        filing: Mapping[str, Any],
        available: datetime,
        collected: datetime,
    ) -> list[AlternativeDataEvent]:
        accession = str(filing.get("accessionNumber", ""))
        document = str(filing.get("primaryDocument", ""))
        if not accession or not document:
            return []
        accession_path = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession_path}/{document}"
        )
        # Archives accepts the same identifying User-Agent; do not set a wrong Host.
        headers = {"User-Agent": self.sec_user_agent, "Accept-Encoding": "gzip, deflate"}
        root = ET.fromstring(self._get_content(url, headers=headers, sec=True))
        owner = self._xml_descendant_text(root, "rptOwnerName")
        events = []
        transactions = [
            node for node in root.iter()
            if node.tag.rsplit("}", 1)[-1] in {
                "nonDerivativeTransaction", "derivativeTransaction"
            }
        ]
        for index, transaction in enumerate(transactions):
            code = self._xml_descendant_text(transaction, "transactionCode") or ""
            shares = _finite_number(
                self._xml_descendant_text(transaction, "transactionShares")
            )
            price = _finite_number(
                self._xml_descendant_text(transaction, "transactionPricePerShare")
            )
            transaction_date = self._xml_descendant_text(
                transaction, "transactionDate"
            )
            events.append(AlternativeDataEvent(
                category="insider_trade",
                source="sec_form4",
                ticker=ticker,
                external_id=f"{accession}:{index}",
                event_time=_as_utc(transaction_date, fallback=available),
                available_at=available,
                collected_at=collected,
                value=shares * price,
                payload={
                    "transaction_type": code,
                    "shares": shares,
                    "price_per_share": price,
                    "owner": owner,
                    "accession_number": accession,
                },
            ))
        return events

    def collect_sec(self, ticker: str) -> tuple[int, int]:
        headers = self._sec_headers()
        ticker = ticker.upper()
        cik = self._load_sec_tickers().get(ticker)
        if cik is None:
            raise AlternativeDataError(f"SEC CIK not found for ticker {ticker}")
        submissions = self._get_json(
            f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
            headers=headers,
            sec=True,
        )
        collected = self.now().astimezone(UTC)
        recent = submissions.get("filings", {}).get("recent", {})
        rows = self._sec_recent_rows(recent)
        max_filings = int(os.getenv("ALT_DATA_SEC_MAX_FILINGS_PER_TICKER", "40"))
        max_form4_documents = int(os.getenv("ALT_DATA_SEC_MAX_FORM4_DOCUMENTS", "5"))
        events: list[AlternativeDataEvent] = []
        form4_documents = 0
        for filing in rows[:max_filings]:
            form = str(filing.get("form", ""))
            filing_date = filing.get("filingDate")
            accepted_raw = filing.get("acceptanceDateTime")
            available = _as_utc(
                accepted_raw or filing_date,
                fallback=collected,
                end_of_day=not accepted_raw,
            )
            event_time = _as_utc(
                filing.get("reportDate") or filing_date, fallback=available
            )
            accession = str(filing.get("accessionNumber", ""))
            events.append(AlternativeDataEvent(
                category="sec_filing",
                source="sec_submissions",
                ticker=ticker,
                external_id=accession,
                event_time=event_time,
                available_at=available,
                collected_at=collected,
                payload={
                    "form": form,
                    "accession_number": accession,
                    "primary_document": filing.get("primaryDocument"),
                    "filing_date": filing_date,
                    "report_date": filing.get("reportDate"),
                },
            ))
            if (
                form.upper().replace(" ", "") in {"4", "4/A"}
                and form4_documents < max_form4_documents
            ):
                events.extend(self._form4_events(
                    ticker, cik, filing, available, collected
                ))
                form4_documents += 1
        return len(events), self.store.add_events(events)

    @staticmethod
    def _frame_column_sum(frame: Any, column: str) -> float:
        try:
            return _finite_number(frame[column].fillna(0).sum())
        except (KeyError, TypeError, AttributeError):
            return 0.0

    @staticmethod
    def _frame_column_values(frame: Any, column: str) -> list[float]:
        try:
            values = frame[column].dropna().tolist()
        except (KeyError, TypeError, AttributeError):
            return []
        return [
            number for value in values
            if math.isfinite(number := _finite_number(value, float("nan")))
        ]

    def collect_yfinance(self, ticker: str) -> tuple[int, int]:
        """Collect news, fundamentals, past earnings, and nearest options."""
        ticker = ticker.upper()
        stock = self.yfinance_factory(ticker)
        collected = self.now().astimezone(UTC)
        events: list[AlternativeDataEvent] = []

        try:
            news_items = stock.get_news(count=20)
        except (AttributeError, TypeError):
            news_items = getattr(stock, "news", [])
        for index, item in enumerate(news_items or []):
            content = item.get("content", item)
            title = str(content.get("title", ""))
            canonical = content.get("canonicalUrl") or {}
            url = (
                canonical.get("url") if isinstance(canonical, Mapping)
                else canonical
            ) or content.get("link") or content.get("url") or ""
            published = _first(
                content, "pubDate", "displayTime", "providerPublishTime",
                default=None,
            )
            event_time = _as_utc(published, fallback=collected)
            events.append(AlternativeDataEvent(
                category="news",
                source="yfinance_news",
                ticker=ticker,
                external_id=str(item.get("id") or url or f"{index}:{title}"),
                event_time=event_time,
                available_at=event_time if published else collected,
                collected_at=collected,
                sentiment=_sentiment(title),
                payload={"title": title, "url": url},
            ))

        info = getattr(stock, "info", {}) or {}
        fundamentals = {
            "market_cap": _finite_number(info.get("marketCap")),
            "pe_ratio": _finite_number(
                info.get("trailingPE", info.get("forwardPE"))
            ),
            "price_to_book": _finite_number(info.get("priceToBook")),
            "revenue_growth": _finite_number(info.get("revenueGrowth")),
            "earnings_growth": _finite_number(info.get("earningsGrowth")),
            "debt_to_equity": _finite_number(info.get("debtToEquity")),
            "free_cashflow": _finite_number(info.get("freeCashflow")),
        }
        events.append(AlternativeDataEvent(
            category="fundamentals",
            source="yfinance_fundamentals",
            ticker=ticker,
            external_id=f"{ticker}:{collected.isoformat()}",
            event_time=collected,
            available_at=collected,
            collected_at=collected,
            payload=fundamentals,
        ))

        try:
            earnings = stock.get_earnings_dates(limit=12)
        except (AttributeError, KeyError, TypeError, ValueError):
            earnings = None
        if earnings is not None and not getattr(earnings, "empty", True):
            for earnings_date, row in earnings.iterrows():
                event_time = _as_utc(earnings_date, fallback=collected)
                if event_time > collected:
                    continue
                surprise = _finite_number(
                    row.get("Surprise(%)", row.get("Surprise %", 0))
                )
                events.append(AlternativeDataEvent(
                    category="earnings",
                    source="yfinance_earnings",
                    ticker=ticker,
                    external_id=f"{ticker}:{event_time.isoformat()}",
                    event_time=event_time,
                    # Historical yfinance revisions are not timestamped.
                    available_at=collected,
                    collected_at=collected,
                    value=surprise,
                    payload={
                        "eps_estimate": _finite_number(row.get("EPS Estimate")),
                        "reported_eps": _finite_number(row.get("Reported EPS")),
                        "surprise_percent": surprise,
                    },
                ))

        expirations = list(getattr(stock, "options", ()) or ())
        if expirations:
            chain = stock.option_chain(expirations[0])
            call_volume = self._frame_column_sum(chain.calls, "volume")
            put_volume = self._frame_column_sum(chain.puts, "volume")
            implied = (
                self._frame_column_values(chain.calls, "impliedVolatility")
                + self._frame_column_values(chain.puts, "impliedVolatility")
            )
            option_payload = {
                "expiration": expirations[0],
                "call_volume": call_volume,
                "put_volume": put_volume,
                "put_call_volume_ratio": (
                    put_volume / call_volume if call_volume > 0 else 0.0
                ),
                "implied_volatility_mean": (
                    sum(implied) / len(implied) if implied else 0.0
                ),
            }
            events.append(AlternativeDataEvent(
                category="options",
                source="yfinance_options",
                ticker=ticker,
                external_id=f"{ticker}:{collected.isoformat()}:{expirations[0]}",
                event_time=collected,
                available_at=collected,
                collected_at=collected,
                payload=option_payload,
            ))
        return len(events), self.store.add_events(events)


def get_point_in_time_features(
    ticker: str,
    as_of: datetime | str,
    lookback_days: int = 30,
    *,
    store: AlternativeDataStore | None = None,
) -> dict[str, Any]:
    """Convenience model-integration API around the default store."""
    return (store or AlternativeDataStore()).get_point_in_time_features(
        ticker, as_of, lookback_days
    )


def collect_all(
    tickers: Sequence[str],
    *,
    store: AlternativeDataStore | None = None,
    sources: set[str] | None = None,
) -> dict[str, Any]:
    """Convenience ingestion API around a default collector."""
    return AlternativeDataCollector(store=store).collect_all(tickers, sources)


__all__ = [
    "AlternativeDataCollector",
    "AlternativeDataError",
    "AlternativeDataEvent",
    "AlternativeDataStore",
    "ConfigurationError",
    "FEATURE_NAMES",
    "SourceStatus",
    "collect_all",
    "get_point_in_time_features",
]
