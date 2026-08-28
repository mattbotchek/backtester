"""Durable prediction, outcome, and model-training history.

Predictions are immutable claims about a future price.  They are stored before
the target time and settled only against a market bar that became available at
or after that time.  This makes the accuracy shown by the application a live
forward metric rather than a cross-validation score.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import exchange_calendars


DEFAULT_DB_PATH = Path(__file__).parent / "models" / "prediction_history.db"
_SCHEMA_LOCK = threading.Lock()


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def to_utc(value: Any) -> datetime:
    """Normalize a datetime-like value to timezone-aware UTC."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def iso_utc(value: Any) -> str:
    """Serialize a datetime-like value as an ISO-8601 UTC timestamp."""
    return to_utc(value).isoformat().replace("+00:00", "Z")


def prediction_target_time(input_at: Any, horizon: int, interval: str) -> datetime:
    """Calculate when the target market bar is complete."""
    timestamp = pd.Timestamp(to_utc(input_at))
    normalized = interval.lower()

    if normalized.endswith("m") and normalized[:-1].isdigit():
        bar_size = pd.Timedelta(minutes=int(normalized[:-1]))
        target_start = _advance_market_bars(timestamp, bar_size, horizon)
        _require_same_session(timestamp, target_start)
        target = target_start + bar_size
    elif normalized.endswith("h") and normalized[:-1].isdigit():
        bar_size = pd.Timedelta(hours=int(normalized[:-1]))
        target_start = _advance_market_bars(timestamp, bar_size, horizon)
        _require_same_session(timestamp, target_start)
        target = target_start + bar_size
    elif normalized in {"1d", "1day", "day"}:
        target = _advance_market_sessions(timestamp, horizon)
    elif normalized.endswith("d") and normalized[:-1].isdigit():
        target = _advance_market_sessions(
            timestamp,
            int(normalized[:-1]) * horizon,
        )
    elif normalized in {"1wk", "1w", "week"}:
        target = _advance_market_sessions(timestamp, 5 * horizon)
    else:
        target = timestamp + pd.Timedelta(days=horizon)

    return to_utc(target)


def _advance_market_bars(
    timestamp: pd.Timestamp,
    bar_size: pd.Timedelta,
    horizon: int,
) -> pd.Timestamp:
    """Advance through regular US equity-session bars."""
    if horizon < 1:
        raise ValueError("Prediction horizon must be at least one bar")
    minutes_per_bar = int(bar_size / pd.Timedelta(minutes=1))
    if minutes_per_bar < 1:
        raise ValueError("Market bar size must be at least one minute")
    calendar = _market_calendar()
    current = timestamp.floor("min")
    if not calendar.is_trading_minute(current):
        current = calendar.minute_to_trading_minute(current, direction="next")
    return calendar.minutes_window(
        current,
        horizon * minutes_per_bar + 1,
    )[-1]


def _advance_market_sessions(timestamp: pd.Timestamp, horizon: int) -> pd.Timestamp:
    """Advance by complete XNYS sessions, including holidays and early closes."""
    if horizon < 1:
        raise ValueError("Prediction horizon must be at least one session")
    calendar = _market_calendar()
    session = calendar.date_to_session(
        timestamp.tz_convert("America/New_York").date(),
        direction="previous",
    )
    target_session = calendar.sessions_window(session, horizon + 1)[-1]
    return calendar.session_close(target_session)


def _require_same_session(
    input_at: pd.Timestamp,
    target_start: pd.Timestamp,
) -> None:
    """Reject intraday forecasts unlike the model's same-session labels."""
    calendar = _market_calendar()
    input_session = calendar.minute_to_session(
        input_at.floor("min"),
        direction="next",
    )
    target_session = calendar.minute_to_session(target_start, direction="none")
    if input_session != target_session:
        raise ValueError(
            "Intraday prediction horizon does not complete in the current "
            "market session"
        )


@lru_cache(maxsize=1)
def _market_calendar():
    return exchange_calendars.get_calendar(
        "XNYS",
        start="1990-01-01",
        end="2100-12-31",
    )


def directional_hit(
    direction: str,
    predicted_price: float,
    actual_price: float,
    flat_tolerance_pct: float = 0.0025,
) -> bool:
    """Apply the product's directional target-price hit rule."""
    normalized = direction.upper()
    if normalized == "UP":
        return actual_price >= predicted_price
    if normalized == "DOWN":
        return actual_price <= predicted_price
    if normalized == "FLAT":
        if predicted_price == 0:
            return actual_price == 0
        return abs(actual_price / predicted_price - 1) <= flat_tolerance_pct
    raise ValueError(f"Unsupported prediction direction: {direction}")


class PredictionStore:
    """SQLite-backed ledger for model training and forward predictions."""

    def __init__(self, path: str | Path | None = None):
        configured_path = path or os.getenv("PREDICTION_DB_PATH") or DEFAULT_DB_PATH
        self.path = Path(configured_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            str(self.path),
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with _SCHEMA_LOCK, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
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

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_trained_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_at TEXT NOT NULL,
                    target_at TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    base_price REAL NOT NULL,
                    predicted_return_pct REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('UP', 'DOWN', 'FLAT')),
                    confidence REAL NOT NULL,
                    actual_price REAL,
                    actual_at TEXT,
                    hit INTEGER CHECK(hit IN (0, 1)),
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'settled')),
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(model_name) REFERENCES models(model_name),
                    UNIQUE(
                        model_name, model_trained_at, ticker, input_at, target_at
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_due
                    ON predictions(status, target_at);
                CREATE INDEX IF NOT EXISTS idx_predictions_history
                    ON predictions(ticker, model_name, created_at DESC);
                """
            )
            self._migrate_prediction_schema(connection)

    @staticmethod
    def _migrate_prediction_schema(connection: sqlite3.Connection) -> None:
        """Upgrade ledgers created before model training versions were keyed."""
        table = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type='table' AND name='predictions'
            """
        ).fetchone()
        if table is None:
            return

        schema = "".join((table["sql"] or "").lower().split())
        expected_key = (
            "unique(model_name,model_trained_at,ticker,input_at,target_at)"
        )
        if expected_key in schema:
            return

        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        }
        trained_at = (
            "COALESCE(model_trained_at, 'unknown')"
            if "model_trained_at" in columns
            else "'unknown'"
        )
        connection.executescript(
            f"""
            ALTER TABLE predictions RENAME TO predictions_legacy;
            DROP INDEX IF EXISTS idx_predictions_due;
            DROP INDEX IF EXISTS idx_predictions_history;

            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_trained_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                created_at TEXT NOT NULL,
                input_at TEXT NOT NULL,
                target_at TEXT NOT NULL,
                interval TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                base_price REAL NOT NULL,
                predicted_return_pct REAL NOT NULL,
                predicted_price REAL NOT NULL,
                direction TEXT NOT NULL
                    CHECK(direction IN ('UP', 'DOWN', 'FLAT')),
                confidence REAL NOT NULL,
                actual_price REAL,
                actual_at TEXT,
                hit INTEGER CHECK(hit IN (0, 1)),
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'settled')),
                metadata_json TEXT NOT NULL DEFAULT '{{}}',
                FOREIGN KEY(model_name) REFERENCES models(model_name),
                UNIQUE(
                    model_name, model_trained_at, ticker, input_at, target_at
                )
            );

            INSERT INTO predictions (
                id, model_name, model_trained_at, ticker, created_at, input_at,
                target_at, interval, horizon, base_price,
                predicted_return_pct, predicted_price, direction, confidence,
                actual_price, actual_at, hit, status, metadata_json
            )
            SELECT
                id, model_name, {trained_at}, ticker, created_at, input_at,
                target_at, interval, horizon, base_price,
                predicted_return_pct, predicted_price, direction, confidence,
                actual_price, actual_at, hit, status, metadata_json
            FROM predictions_legacy;

            DROP TABLE predictions_legacy;
            CREATE INDEX idx_predictions_due
                ON predictions(status, target_at);
            CREATE INDEX idx_predictions_history
                ON predictions(ticker, model_name, created_at DESC);
            """
        )

    def register_model(
        self,
        model_name: str,
        family: str,
        horizon: int,
        interval: str,
        last_trained_at: Any | None = None,
        training_metrics: dict[str, Any] | None = None,
        target_accuracy: float = 0.70,
        minimum_accuracy: float = 0.65,
    ) -> None:
        """Create or update the durable status row for a model."""
        trained_at = iso_utc(last_trained_at) if last_trained_at else None
        now = iso_utc(utc_now())
        metrics_json = json.dumps(
            _json_safe(training_metrics or {}),
            default=_json_default,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO models (
                    model_name, family, horizon, interval, last_trained_at,
                    training_metrics_json, target_accuracy, minimum_accuracy,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_name) DO UPDATE SET
                    family=excluded.family,
                    horizon=excluded.horizon,
                    interval=excluded.interval,
                    last_trained_at=COALESCE(
                        excluded.last_trained_at, models.last_trained_at
                    ),
                    training_metrics_json=CASE
                        WHEN excluded.last_trained_at IS NOT NULL
                        THEN excluded.training_metrics_json
                        ELSE models.training_metrics_json
                    END,
                    target_accuracy=excluded.target_accuracy,
                    minimum_accuracy=excluded.minimum_accuracy,
                    updated_at=excluded.updated_at
                """,
                (
                    model_name,
                    family,
                    int(horizon),
                    interval,
                    trained_at,
                    metrics_json,
                    float(target_accuracy),
                    float(minimum_accuracy),
                    now,
                ),
            )

    def record_prediction(
        self,
        *,
        model_name: str,
        family: str,
        ticker: str,
        input_at: Any,
        target_at: Any,
        interval: str,
        horizon: int,
        base_price: float,
        predicted_return_pct: float,
        direction: str,
        confidence: float,
        created_at: Any | None = None,
        last_trained_at: Any | None = None,
        training_metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one prediction and return the canonical stored record."""
        normalized_direction = direction.upper()
        if normalized_direction not in {"UP", "DOWN", "FLAT"}:
            raise ValueError(f"Unsupported prediction direction: {direction}")
        if base_price <= 0:
            raise ValueError("base_price must be positive")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

        self.register_model(
            model_name=model_name,
            family=family,
            horizon=horizon,
            interval=interval,
            last_trained_at=last_trained_at,
            training_metrics=training_metrics,
        )

        input_iso = iso_utc(input_at)
        target_iso = iso_utc(target_at)
        created_iso = iso_utc(created_at or utc_now())
        trained_iso = (
            iso_utc(last_trained_at)
            if last_trained_at
            else "unknown"
        )
        predicted_price = base_price * (1 + predicted_return_pct / 100)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    model_name, model_trained_at, ticker, created_at, input_at,
                    target_at, interval, horizon, base_price,
                    predicted_return_pct, predicted_price, direction,
                    confidence, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    model_name, model_trained_at, ticker, input_at, target_at
                ) DO NOTHING
                """,
                (
                    model_name,
                    trained_iso,
                    ticker.upper(),
                    created_iso,
                    input_iso,
                    target_iso,
                    interval,
                    int(horizon),
                    float(base_price),
                    float(predicted_return_pct),
                    float(predicted_price),
                    normalized_direction,
                    float(confidence),
                    json.dumps(_json_safe(metadata or {}), default=_json_default),
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM predictions
                WHERE model_name=? AND model_trained_at=? AND ticker=?
                    AND input_at=? AND target_at=?
                """,
                (
                    model_name,
                    trained_iso,
                    ticker.upper(),
                    input_iso,
                    target_iso,
                ),
            ).fetchone()
        if row is None:
            raise RuntimeError("Prediction could not be persisted")
        return _prediction_dict(row)

    def settle_prediction(
        self,
        prediction_id: int,
        actual_price: float,
        actual_at: Any,
    ) -> dict[str, Any]:
        """Settle one pending prediction against its realized market price."""
        if actual_price <= 0:
            raise ValueError("actual_price must be positive")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE id=?",
                (prediction_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown prediction id: {prediction_id}")
            if row["status"] == "settled":
                return _prediction_dict(row)

            hit = directional_hit(
                row["direction"],
                float(row["predicted_price"]),
                float(actual_price),
            )
            connection.execute(
                """
                UPDATE predictions
                SET actual_price=?, actual_at=?, hit=?, status='settled'
                WHERE id=? AND status='pending'
                """,
                (
                    float(actual_price),
                    iso_utc(actual_at),
                    int(hit),
                    prediction_id,
                ),
            )
            settled = connection.execute(
                "SELECT * FROM predictions WHERE id=?",
                (prediction_id,),
            ).fetchone()
        return _prediction_dict(settled)

    def settle_due(
        self,
        *,
        now: Any | None = None,
        price_fetcher: Callable[[str, datetime, str], Iterable[tuple[Any, float]]]
        | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Settle due predictions using the first observed bar at target time."""
        cutoff = to_utc(now or utc_now())
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM predictions
                WHERE status='pending' AND target_at <= ?
                ORDER BY target_at
                LIMIT ?
                """,
                (iso_utc(cutoff), int(limit)),
            ).fetchall()

        fetcher = price_fetcher or _fetch_market_prices
        settled = 0
        unavailable = 0
        errors: list[str] = []
        observation_cache: dict[
            tuple[str, str, str],
            Iterable[tuple[Any, float]],
        ] = {}

        for row in rows:
            target_at = to_utc(row["target_at"])
            try:
                cache_key = (
                    row["ticker"],
                    row["interval"],
                    target_at.date().isoformat(),
                )
                if price_fetcher is not None:
                    observations = list(fetcher(
                        row["ticker"],
                        target_at,
                        row["interval"],
                    ))
                elif cache_key not in observation_cache:
                    observation_cache[cache_key] = list(fetcher(
                        row["ticker"],
                        target_at,
                        row["interval"],
                    ))
                    observations = observation_cache[cache_key]
                else:
                    observations = observation_cache[cache_key]
                deadline = target_at + timedelta(days=8)
                eligible = [
                    (to_utc(observed_at), float(price))
                    for observed_at, price in observations
                    if target_at <= to_utc(observed_at) <= deadline
                ]
                if not eligible:
                    unavailable += 1
                    continue
                actual_at, actual_price = min(eligible, key=lambda item: item[0])
                self.settle_prediction(row["id"], actual_price, actual_at)
                settled += 1
            except Exception as exc:
                errors.append(f"{row['ticker']}#{row['id']}: {exc}")

        return {
            "due": len(rows),
            "settled": settled,
            "unavailable": unavailable,
            "errors": errors,
        }

    def get_history(
        self,
        *,
        ticker: str | None = None,
        model_name: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return newest predictions, optionally filtered by ticker/model."""
        clauses: list[str] = []
        parameters: list[Any] = []
        if ticker:
            clauses.append("ticker=?")
            parameters.append(ticker.upper())
        if model_name:
            clauses.append("model_name=?")
            parameters.append(model_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(int(limit))

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM predictions
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [_prediction_dict(row) for row in rows]

    def model_statistics(self, model_name: str | None = None) -> list[dict[str, Any]]:
        """Return live forward-accuracy and training status for each model."""
        parameters: tuple[Any, ...] = ()
        where = ""
        if model_name:
            where = "WHERE m.model_name=?"
            parameters = (model_name,)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    m.*,
                    COUNT(p.id) AS prediction_count,
                    SUM(CASE WHEN p.status='pending' THEN 1 ELSE 0 END)
                        AS pending_count,
                    SUM(CASE WHEN p.status='settled' THEN 1 ELSE 0 END)
                        AS settled_count,
                    SUM(CASE WHEN p.hit=1 THEN 1 ELSE 0 END) AS hit_count,
                    AVG(CASE WHEN p.status='settled' THEN p.hit END)
                        AS forward_accuracy,
                    MAX(p.created_at) AS last_prediction_at
                FROM models m
                LEFT JOIN predictions p ON p.model_name=m.model_name
                {where}
                GROUP BY m.model_name
                ORDER BY m.model_name
                """,
                parameters,
            ).fetchall()

        statistics = []
        for row in rows:
            item = dict(row)
            item["training_metrics"] = json.loads(
                item.pop("training_metrics_json") or "{}"
            )
            item["prediction_count"] = int(item["prediction_count"] or 0)
            item["pending_count"] = int(item["pending_count"] or 0)
            item["settled_count"] = int(item["settled_count"] or 0)
            item["hit_count"] = int(item["hit_count"] or 0)
            statistics.append(item)
        return statistics

    def current_prediction(
        self,
        ticker: str,
        model_name: str,
    ) -> dict[str, Any] | None:
        """Return the latest prediction for one ticker/model pair."""
        history = self.get_history(
            ticker=ticker,
            model_name=model_name,
            limit=1,
        )
        return history[0] if history else None


def _fetch_market_prices(
    ticker: str,
    target_at: datetime,
    interval: str,
) -> list[tuple[Any, float]]:
    """Fetch a small market window used only to settle due predictions."""
    import yfinance

    intraday = interval.endswith("m") or interval.endswith("h")
    if intraday:
        start = target_at.date().isoformat()
        window_days = 7 if interval == "1m" else 9
        end = (target_at + timedelta(days=window_days)).date().isoformat()
        history = yfinance.Ticker(ticker).history(
            start=start,
            end=end,
            interval=interval,
        )
    else:
        start = (target_at - timedelta(days=7)).date().isoformat()
        end = (target_at + timedelta(days=9)).date().isoformat()
        history = yfinance.Ticker(ticker).history(
            start=start,
            end=end,
            interval=interval,
        )
    if history.empty:
        return []
    return [
        (_bar_completion_time(timestamp, interval), float(close))
        for timestamp, close in zip(
            history.index,
            history["Close"],
            strict=True,
        )
    ]


def _bar_completion_time(timestamp: Any, interval: str) -> datetime:
    """Normalize provider bar-start labels to their completion timestamps."""
    start = pd.Timestamp(to_utc(timestamp))
    normalized = interval.lower()
    if normalized.endswith("m") and normalized[:-1].isdigit():
        return to_utc(start + pd.Timedelta(minutes=int(normalized[:-1])))
    if normalized.endswith("h") and normalized[:-1].isdigit():
        return to_utc(start + pd.Timedelta(hours=int(normalized[:-1])))

    calendar = _market_calendar()
    local_date = start.tz_convert("America/New_York").date()
    if normalized in {"1wk", "1w", "week"}:
        local_date += timedelta(days=6)
    session = calendar.date_to_session(local_date, direction="previous")
    return to_utc(calendar.session_close(session))


def _prediction_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["hit"] = None if item["hit"] is None else bool(item["hit"])
    item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
    return item


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return iso_utc(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
