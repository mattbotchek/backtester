"""Stock Predictor Web Application.

FastAPI-based web server providing:
  - Dashboard with most popular stock predictions
  - Per-stock analysis with strategies and ML predictions
  - Real-time charts with prediction overlays
  - Your Picks page for user-selected stocks
  - Market Wide predictions page
  - 24/7 training pipeline with progress tracking
  - Day trading opt-in per stock

Run with: python -m webapp.app
"""

import asyncio
import logging
import os
import sys
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance
from fastapi import FastAPI, Request, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies import get_registry
from ml.prediction_store import PredictionStore, iso_utc, utc_now
from webapp.cache import PersistentTTLCache

logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Predictor", version="2.0")

WEBAPP_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))


# ── Ticker Pools ──

# 20 most popular tickers shown on the dashboard
POPULAR_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "LLY", "MA", "AVGO", "HD", "PG",
    "NFLX", "AMD", "CRM", "XOM",
]

# Broad universe of tickers across all sectors/market caps.
# The search endpoint also accepts *any* ticker the user types —
# this list is only for pre-populated suggestions and batch jobs.
KNOWN_TICKERS = sorted(set(POPULAR_TICKERS + [
    # ── S&P 500 (large-cap) ──
    "JNJ", "DIS", "PYPL", "ADBE", "INTC", "CSCO", "PEP", "KO", "MRK",
    "ABT", "CVX", "WMT", "COST", "TGT", "LOW", "NKE", "SBUX", "BA",
    "CAT", "DE", "GE", "MMM", "HON", "UPS", "GS", "MS", "C", "BAC",
    "WFC", "AXP", "PFE", "ABBV", "BMY", "GILD", "TMO", "DHR", "NEE",
    "DUK", "SO", "AEP", "SRE", "AMT", "PLD", "SPG", "O", "QCOM",
    "TXN", "MU", "AMAT", "COP", "EOG", "SLB", "PSX", "MCO", "SPGI",
    "CME", "ICE", "SCHW", "BLK", "T", "VZ", "TMUS", "CMCSA", "CHTR",
    "FDX", "UNP", "CSX", "NSC", "LIN", "APD", "SHW", "ECL", "DD",
    "EMR", "ETN", "ROK", "ISRG", "SYK", "MDT", "ZTS", "DXCM", "IDXX",
    "NOW", "SNOW", "UBER", "ABNB", "SQ", "SHOP", "PLTR", "PANW",
    "CRWD", "DDOG", "ZS", "NET", "TEAM", "MDB", "COIN", "MELI",
    "SE", "SPOT", "SNAP", "PINS", "RBLX", "RIVN", "LCID", "NIO",
    # ── Mid-cap / growth ──
    "ENPH", "SEDG", "FSLR", "PLUG", "RUN", "CHPT", "SOFI", "AFRM",
    "HOOD", "UPST", "DKNG", "PENN", "WYNN", "LVS", "MGM", "MAR",
    "HLT", "DAL", "UAL", "LUV", "AAL", "CCL", "RCL", "NCLH",
    # ── Small-cap / speculative ──
    "GME", "AMC", "BB", "BBBY", "CLOV", "WISH", "PLBY",
    "MARA", "RIOT", "HUT", "BITF", "CLSK",
    # ── ETFs (sector / thematic) ──
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "ARKK", "ARKG",
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY",
    "GLD", "SLV", "USO", "TLT", "HYG", "LQD", "IBIT", "BITO",
    # ── International ADRs ──
    "TSM", "BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV",
    "SAP", "TM", "SONY", "MFG", "UL", "GSK", "AZN", "NVO", "ASML",
]))

# ── Timeframe mapping: label → yfinance period string ──
TIMEFRAMES = {
    "1mo": "1mo",
    "3mo": "3mo",
    "6mo": "6mo",
    "ytd": "ytd",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "max": "max",
}
DEFAULT_TIMEFRAME = "ytd"


# ── Training progress tracking ──

class TrainingTracker:
    """Thread-safe tracker for model training progress."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status: dict[str, dict] = {}

    def start(self, model_name: str, total_tickers: int = 0):
        with self._lock:
            self._status[model_name] = {
                "state": "training",
                "progress": 0,
                "total": total_tickers,
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "error": None,
                "accuracy": None,
            }

    def update(self, model_name: str, progress: int):
        with self._lock:
            if model_name in self._status:
                self._status[model_name]["progress"] = progress

    def finish(self, model_name: str, accuracy: float | None = None):
        with self._lock:
            if model_name in self._status:
                self._status[model_name]["state"] = "ready"
                self._status[model_name]["progress"] = self._status[model_name]["total"]
                self._status[model_name]["finished_at"] = datetime.now().isoformat()
                self._status[model_name]["accuracy"] = accuracy

    def fail(self, model_name: str, error: str):
        with self._lock:
            if model_name in self._status:
                self._status[model_name]["state"] = "error"
                self._status[model_name]["error"] = error

    def get_all(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._status)

    def get(self, model_name: str) -> dict | None:
        with self._lock:
            return self._status.get(model_name)


training_tracker = TrainingTracker()

_settlement_lock = threading.Lock()
_last_settlement_at: Optional[datetime] = None
_settlement_in_progress = False
_web_cache = PersistentTTLCache()
_prediction_refresh_lock = threading.Lock()
_prediction_refreshes: set[str] = set()
_prediction_refresh_slots = threading.Semaphore(2)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Expose server duration and flag handlers that violate the page budget."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.1f}"
    if elapsed_ms >= 5000:
        logger.warning(
            "Slow request %s %s completed in %.2fs",
            request.method,
            request.url.path,
            elapsed_ms / 1000,
        )
    return response


# ── Helper functions ──

def _get_stock_history(ticker: str, period: str = "1y") -> dict:
    """Get price history and quote statistics from yfinance."""
    hist = yfinance.Ticker(ticker).history(period=period)

    if hist.empty:
        raise ValueError(f"No data for {ticker}")

    return {
        "ticker": ticker,
        "current_price": float(hist["Close"].iloc[-1]),
        "change_1d": float(hist["Close"].pct_change().iloc[-1] * 100),
        "change_5d": float(
            hist["Close"].pct_change(5).iloc[-1] * 100
        ) if len(hist) > 5 else 0,
        "change_20d": float(
            hist["Close"].pct_change(20).iloc[-1] * 100
        ) if len(hist) > 20 else 0,
        "volume": int(hist["Volume"].iloc[-1]),
        "high_52w": float(hist["Close"].max()),
        "low_52w": float(hist["Close"].min()),
        "history": {
            "dates": [d.strftime("%Y-%m-%d") for d in hist.index[-60:]],
            "close": [round(float(v), 2) for v in hist["Close"].iloc[-60:]],
            "volume": [int(v) for v in hist["Volume"].iloc[-60:]],
        },
    }


def _get_stock_metadata(ticker: str) -> dict:
    """Get slowly changing company metadata independently of live quotes."""
    info = yfinance.Ticker(ticker).info
    if not info:
        raise ValueError(f"No metadata for {ticker}")
    return {
        "name": info.get("longName", info.get("shortName", ticker)),
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE"),
    }


def _cached_payload(
    key: str,
    loader,
    *,
    ttl_seconds: float,
    fallback: dict,
    wait_seconds: float = 3.5,
) -> dict:
    """Load JSON-ready data without allowing remote sources to stall a page."""
    value, state = _web_cache.get(
        key,
        loader,
        ttl_seconds=ttl_seconds,
        wait_seconds=wait_seconds,
        fallback=fallback,
    )
    payload = deepcopy(value)
    if isinstance(payload, dict):
        payload["cache_state"] = state
    return payload


def _run_strategies(ticker: str, timeline: str = "ytd") -> list[dict]:
    """Run all strategies on a ticker with the given timeframe."""
    results = []
    registry = get_registry()
    yf_period = TIMEFRAMES.get(timeline, timeline)

    for _name, strategy_cls in registry.items():
        try:
            strategy = strategy_cls(
                timeline=yf_period, interval="1d", ticker=ticker, cash=10000,
            )
            result = strategy.calculate()
            results.append({
                "name": strategy.name,
                "is_novel": getattr(strategy, "is_novel", False),
                "total_return": round(result.roi_percent, 2),
                "win_rate": round(result.win_rate * 100, 1),
                "num_trades": result.total_trades,
                "avg_return": round(
                    result.roi_percent / max(result.total_trades, 1), 2
                ),
            })
        except Exception:
            pass

    return sorted(results, key=lambda x: x["total_return"], reverse=True)


def _get_ml_prediction(ticker: str) -> dict:
    """Get ML prediction for a ticker."""
    try:
        from ml.model import StockPredictor

        _settle_due_predictions()
        predictor = StockPredictor()
        predictor.load()
        pred = predictor.predict(ticker)
        return {
            "model_name": pred.model_name,
            "direction": pred.direction,
            "predicted_return": round(pred.predicted_return_pct, 2),
            "confidence": round(pred.confidence * 100, 1),
            "outliers": pred.outliers,
            "created_at": pred.timestamp,
            "input_at": pred.input_at,
            "target_at": pred.predicted_for,
            "base_price": round(pred.base_price, 4),
            "predicted_price": round(pred.predicted_price, 4),
            "last_trained_at": pred.last_trained_at,
            "forward_accuracy": (
                round(pred.forward_accuracy * 100, 1)
                if pred.forward_accuracy is not None
                else None
            ),
            "settled_predictions": pred.settled_predictions,
            "interval": "1d",
            "horizon": predictor.horizon,
        }
    except Exception as e:
        logger.debug("Prediction failed for %s: %s", ticker, e)
        return {
            "direction": "N/A",
            "predicted_return": 0,
            "confidence": 0,
            "outliers": [],
            "model_status": "not_available",
        }


def _settle_due_predictions(force: bool = False) -> dict:
    """Throttle market lookups while keeping realized outcomes current."""
    global _last_settlement_at, _settlement_in_progress

    if (
        not force
        and os.environ.get("WEBAPP_CONTINUOUS_PIPELINE", "0") == "1"
    ):
        return {"status": "coordinated_pipeline"}

    with _settlement_lock:
        now = utc_now()
        if _settlement_in_progress:
            return {"status": "in_progress"}
        if (
            not force
            and _last_settlement_at
            and now - _last_settlement_at < timedelta(minutes=5)
        ):
            return {"status": "throttled"}
        _last_settlement_at = now
        _settlement_in_progress = True

    def settle() -> dict:
        global _settlement_in_progress

        try:
            return PredictionStore().settle_due(now=now)
        finally:
            with _settlement_lock:
                _settlement_in_progress = False

    if force:
        return settle()

    def settle_in_background() -> None:
        try:
            settle()
        except Exception:
            logger.exception("Background prediction settlement failed")

    threading.Thread(target=settle_in_background, daemon=True).start()
    return {"status": "started"}


def _normalize_model_prediction(prediction: dict) -> dict:
    """Normalize model-native values to the public API representation."""
    normalized = dict(prediction)
    normalized["predicted_return"] = round(
        float(normalized.get("predicted_return", 0)),
        3,
    )
    normalized["confidence"] = round(
        float(normalized.get("confidence", 0)) * 100,
        1,
    )
    if normalized.get("forward_accuracy") is not None:
        normalized["forward_accuracy"] = round(
            float(normalized["forward_accuracy"]) * 100,
            1,
        )
    for field in ("base_price", "predicted_price"):
        if normalized.get(field) is not None:
            normalized[field] = round(float(normalized[field]), 4)
    return normalized


def _stored_prediction(
    prediction: dict,
    model_statistics: dict[str, dict],
) -> dict:
    """Normalize a durable ledger prediction to the public API shape."""
    model = model_statistics.get(prediction["model_name"], {})
    metadata = prediction.get("metadata") or {}
    forward_accuracy = model.get("forward_accuracy")
    return {
        "model_name": prediction["model_name"],
        "ticker": prediction["ticker"],
        "direction": prediction["direction"],
        "predicted_return": round(
            float(prediction["predicted_return_pct"]),
            3,
        ),
        "confidence": round(float(prediction["confidence"]) * 100, 1),
        "outliers": metadata.get("outliers", []),
        "created_at": prediction["created_at"],
        "input_at": prediction["input_at"],
        "target_at": prediction["target_at"],
        "base_price": round(float(prediction["base_price"]), 4),
        "predicted_price": round(float(prediction["predicted_price"]), 4),
        "last_trained_at": model.get(
            "last_trained_at",
            prediction["model_trained_at"],
        ),
        "forward_accuracy": (
            round(float(forward_accuracy) * 100, 1)
            if forward_accuracy is not None
            else None
        ),
        "settled_predictions": int(model.get("settled_count", 0)),
        "interval": prediction["interval"],
        "horizon": int(prediction["horizon"]),
        "status": prediction["status"],
        "actual_price": prediction.get("actual_price"),
        "actual_at": prediction.get("actual_at"),
        "hit": prediction.get("hit"),
    }


def _stored_predictions(
    tickers: list[str],
    *,
    model_name: str | None = None,
) -> list[dict]:
    store = PredictionStore()
    statistics = {
        item["model_name"]: item
        for item in store.model_statistics()
    }
    return [
        _stored_prediction(item, statistics)
        for item in store.latest_predictions(
            tickers=tickers,
            model_name=model_name,
        )
    ]


def _schedule_prediction_refresh(ticker: str) -> bool:
    """Generate missing on-demand predictions without blocking a web request."""
    normalized = ticker.upper()
    with _prediction_refresh_lock:
        if normalized in _prediction_refreshes:
            return True
        if len(_prediction_refreshes) >= 20:
            return False
        _prediction_refreshes.add(normalized)

    def refresh() -> None:
        try:
            with _prediction_refresh_slots:
                _get_all_ml_predictions(normalized)
        except Exception:
            logger.exception("On-demand prediction refresh failed for %s", normalized)
        finally:
            with _prediction_refresh_lock:
                _prediction_refreshes.discard(normalized)

    threading.Thread(
        target=refresh,
        name=f"prediction-refresh-{normalized}",
        daemon=True,
    ).start()
    return True


def _get_all_ml_predictions(ticker: str) -> tuple[list[dict], list[dict]]:
    """Generate current predictions from every trained model family."""
    from ml.multi_horizon import HORIZONS, HorizonModel, MODEL_DIR

    _settle_due_predictions()
    jobs = []

    default_path = MODEL_DIR / "stock_predictor.pkl"
    if default_path.exists():
        jobs.append(("stock_predictor", "default"))

    for name, config in HORIZONS.items():
        if (MODEL_DIR / f"{config.model_file}.pkl").exists():
            jobs.append((name, "horizon"))

    if (MODEL_DIR / "combined_model.pkl").exists():
        jobs.append(("combined_model", "combined"))

    predictions: list[dict] = []
    errors: list[dict] = []

    def predict(job: tuple[str, str]) -> dict:
        name, family = job
        if family == "default":
            result = _get_ml_prediction(ticker)
            if result.get("direction") == "N/A":
                raise RuntimeError("Default model prediction unavailable")
            return result
        if family == "horizon":
            model = HorizonModel(HORIZONS[name])
            model.load()
            return _normalize_model_prediction(model.predict_ticker(ticker))

        from ml.sentiment.combined_model import CombinedPredictor

        model = CombinedPredictor()
        model.load()
        return _normalize_model_prediction(model.predict(ticker))

    worker_count = min(4, max(1, len(jobs)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_jobs = {executor.submit(predict, job): job for job in jobs}
        for future in as_completed(future_jobs):
            name, family = future_jobs[future]
            try:
                predictions.append(future.result())
            except Exception as exc:
                logger.warning("%s prediction failed for %s: %s", name, ticker, exc)
                errors.append({"model_name": name, "family": family, "error": str(exc)})

    predictions.sort(key=lambda item: (item.get("horizon", 0), item["model_name"]))
    return predictions, errors


def _auto_train_models():
    """Background thread that trains the default model on server start."""
    try:
        from ml.model import StockPredictor

        model_name = "default"
        tickers = POPULAR_TICKERS[:20]
        training_tracker.start(model_name, total_tickers=len(tickers))
        logger.info("Auto-training default model on %d tickers...", len(tickers))

        predictor = StockPredictor()
        stats = predictor.train(tickers, timeline="5y")
        predictor.save()

        accuracy = stats.get("cv_accuracy_mean", 0)
        training_tracker.finish(model_name, accuracy=accuracy)
        logger.info("Auto-training complete: CV accuracy=%.3f", accuracy)
    except Exception as e:
        training_tracker.fail("default", str(e))
        logger.error("Auto-training failed: %s", e)


# ── Page routes ──

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard — overview of market and top predictions."""
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_detail(request: Request, ticker: str):
    """Per-stock detail page with strategies and predictions."""
    return templates.TemplateResponse(
        request, "stock_detail.html", {"ticker": ticker.upper()}
    )


@app.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request):
    """Strategies overview page."""
    return templates.TemplateResponse(request, "strategies.html")


@app.get("/ml", response_class=HTMLResponse)
async def ml_page(request: Request):
    """ML model control and status page."""
    return templates.TemplateResponse(request, "ml_control.html")


@app.get("/picks", response_class=HTMLResponse)
async def your_picks_page(request: Request):
    """Your Picks — user-selected stocks page."""
    return templates.TemplateResponse(request, "your_picks.html")


@app.get("/market", response_class=HTMLResponse)
async def market_wide_page(request: Request):
    """Market-wide predictions page."""
    return templates.TemplateResponse(request, "market_wide.html")


# ── API Endpoints ──

@app.get("/api/tickers")
async def api_tickers(q: str = Query("", description="Search query")):
    """Search tickers. Any ticker the user types is valid — this list
    provides autocompletion suggestions from the known pool."""
    upper_q = q.upper().strip()
    if upper_q:
        matches = [t for t in KNOWN_TICKERS if upper_q in t]
        # If the query is an exact ticker not in KNOWN_TICKERS, include it
        if upper_q not in matches and len(upper_q) <= 5 and upper_q.isalpha():
            matches.insert(0, upper_q)
        return {"tickers": matches[:20]}
    return {"tickers": POPULAR_TICKERS}


@app.get("/api/popular")
async def api_popular():
    """Get the 20 most popular tickers with quick stats."""
    def load() -> dict:
        results = []
        for ticker in POPULAR_TICKERS:
            try:
                hist = yfinance.Ticker(ticker).history(period="5d")
                if hist.empty:
                    continue
                price = float(hist["Close"].iloc[-1])
                change = float(hist["Close"].pct_change().iloc[-1] * 100)
                results.append({
                    "ticker": ticker,
                    "price": round(price, 2),
                    "change": round(change, 2),
                })
            except Exception:
                logger.debug("Popular quote unavailable for %s", ticker)
        return {"popular": results}

    return await run_in_threadpool(
        _cached_payload,
        "popular",
        load,
        ttl_seconds=300,
        fallback={"popular": []},
    )


@app.get("/api/stock/{ticker}")
async def api_stock(ticker: str):
    """Get stock data and quick stats."""
    normalized = ticker.upper()
    history_call = run_in_threadpool(
        _cached_payload,
        f"stock-history:{normalized}:1y",
        lambda: _get_stock_history(normalized),
        ttl_seconds=300,
        fallback={"ticker": normalized, "error": "Market data is refreshing"},
    )
    metadata_call = run_in_threadpool(
        _cached_payload,
        f"stock-metadata:{normalized}",
        lambda: _get_stock_metadata(normalized),
        ttl_seconds=86400,
        wait_seconds=0.1,
        fallback={
            "name": normalized,
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": 0,
            "pe_ratio": None,
        },
    )
    history, metadata = await asyncio.gather(history_call, metadata_call)
    states = {history.pop("cache_state"), metadata.pop("cache_state")}
    payload = {**metadata, **history}
    payload["cache_state"] = (
        "refreshing"
        if states & {"refreshing", "error"}
        else "stale"
        if "stale" in states
        else "fresh"
        if states == {"fresh"}
        else "loaded"
    )
    return JSONResponse(payload)


@app.get("/api/stock/{ticker}/strategies")
async def api_strategies(
    ticker: str,
    timeline: str = Query(DEFAULT_TIMEFRAME, description="Backtest timeframe"),
):
    """Run all strategies on a stock with a given timeframe."""
    normalized = ticker.upper()
    payload = await run_in_threadpool(
        _cached_payload,
        f"strategies:{normalized}:{timeline}",
        lambda: {
            "ticker": normalized,
            "timeline": timeline,
            "strategies": _run_strategies(normalized, timeline),
        },
        ttl_seconds=900,
        fallback={
            "ticker": normalized,
            "timeline": timeline,
            "strategies": [],
        },
    )
    results = payload["strategies"]
    return {
        "ticker": normalized,
        "timeline": timeline,
        "strategies": results,
        "cache_state": payload["cache_state"],
    }


@app.get("/api/stock/{ticker}/prediction")
async def api_prediction(ticker: str):
    """Get the default ML prediction for a stock."""
    normalized = ticker.upper()
    predictions = _stored_predictions(
        [normalized],
        model_name="stock_predictor",
    )
    refreshing = not predictions and _schedule_prediction_refresh(normalized)
    pred = predictions[0] if predictions else {
        "direction": "N/A",
        "predicted_return": 0,
        "confidence": 0,
        "model_status": "refreshing" if refreshing else "not_available",
    }
    return {
        "ticker": normalized,
        "prediction": pred,
        "refreshing": refreshing,
    }


@app.get("/api/stock/{ticker}/predictions")
async def api_all_predictions(ticker: str):
    """Get current predictions and realized history from every trained model."""
    normalized_ticker = ticker.upper()
    store = PredictionStore()
    predictions = _stored_predictions([normalized_ticker])
    registered_models = store.model_statistics()
    refreshing = (
        len(predictions) < len(registered_models)
        and _schedule_prediction_refresh(normalized_ticker)
    )
    history = store.get_history(ticker=normalized_ticker, limit=500)
    return {
        "ticker": normalized_ticker,
        "predictions": predictions,
        "history": history,
        "errors": [],
        "refreshing": refreshing,
    }


@app.get("/api/predictions/history")
async def api_prediction_history(
    ticker: str | None = Query(None),
    model: str | None = Query(None),
    limit: int = Query(200, ge=1, le=2000),
):
    """Query the auditable prediction/outcome ledger."""
    _settle_due_predictions()
    return {
        "predictions": PredictionStore().get_history(
            ticker=ticker,
            model_name=model,
            limit=limit,
        )
    }


@app.post("/api/predictions/settle")
async def api_settle_predictions():
    """Force a refresh of all due prediction outcomes."""
    return _settle_due_predictions(force=True)


@app.get("/api/models/status")
async def api_model_status():
    """Get last-trained times and live forward accuracy for every model."""
    _settle_due_predictions()
    return {"models": PredictionStore().model_statistics()}


@app.get("/api/market/overview")
async def api_market_overview():
    """Get market overview (major indices + top movers)."""
    def load() -> dict:
        indices = {}
        for symbol, name in [
            ("SPY", "S&P 500"),
            ("QQQ", "Nasdaq 100"),
            ("IWM", "Russell 2000"),
            ("DIA", "Dow Jones"),
            ("^VIX", "VIX"),
        ]:
            try:
                data = yfinance.Ticker(symbol).history(period="5d")
                if not data.empty and len(data) >= 2:
                    indices[name] = {
                        "price": round(float(data["Close"].iloc[-1]), 2),
                        "change": round(
                            float(data["Close"].pct_change().iloc[-1] * 100), 2
                        ),
                    }
            except Exception:
                logger.debug("Market overview quote unavailable for %s", symbol)
        return {"indices": indices}

    return await run_in_threadpool(
        _cached_payload,
        "market-overview",
        load,
        ttl_seconds=60,
        fallback={"indices": {}},
    )


@app.get("/api/market/predictions")
async def api_market_predictions():
    """Get market-wide predictions: overall direction and top movers."""
    predictions = _stored_predictions(
        POPULAR_TICKERS,
        model_name="stock_predictor",
    )

    # Sort by absolute predicted return for "highest movers"
    predictions.sort(key=lambda x: abs(x["predicted_return"]), reverse=True)

    # Overall market sentiment
    up_count = sum(1 for p in predictions if p["direction"] == "UP")
    down_count = sum(1 for p in predictions if p["direction"] == "DOWN")
    total = len(predictions)
    if total > 0:
        market_direction = "BULLISH" if up_count > down_count else "BEARISH"
        market_confidence = round(max(up_count, down_count) / total * 100, 1)
    else:
        market_direction = "N/A"
        market_confidence = 0

    return {
        "market": {
            "direction": market_direction,
            "confidence": market_confidence,
            "up_count": up_count,
            "down_count": down_count,
            "total": total,
        },
        "top_movers_up": [p for p in predictions if p["direction"] == "UP"][:10],
        "top_movers_down": [
            p for p in predictions if p["direction"] == "DOWN"
        ][:10],
        "all_predictions": predictions,
    }


@app.get("/api/predictions/daily")
async def api_daily_predictions():
    """Get daily predictions for popular stocks."""
    stored = _stored_predictions(
        POPULAR_TICKERS,
        model_name="stock_predictor",
    )
    predictions = {prediction["ticker"]: prediction for prediction in stored}
    return {"predictions": predictions, "cached": True}


@app.get("/api/training/status")
async def api_training_status():
    """Get current training progress for all models."""
    return {
        "models": training_tracker.get_all(),
        "forward_models": PredictionStore().model_statistics(),
    }


@app.post("/api/training/start")
async def api_training_start(model: str = Query("default")):
    """Trigger model training (non-blocking)."""
    current = training_tracker.get(model)
    if current and current["state"] == "training":
        return {"status": "already_training", "model": model}

    thread = threading.Thread(target=_auto_train_models, daemon=True)
    thread.start()
    return {"status": "started", "model": model}


@app.get("/api/sentiment")
async def api_sentiment():
    """Get current market sentiment data."""
    try:
        from ml.sentiment.scrapers import get_aggregated_sentiment

        sentiment = get_aggregated_sentiment()
        return {"sentiment": sentiment}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/stock/{ticker}/intraday")
async def api_intraday(
    ticker: str,
    period: str = Query("5d", description="Data period"),
    interval: str = Query("5m", description="Bar interval"),
):
    """Get intraday OHLCV data for candlestick charts."""
    normalized = ticker.upper()

    def load() -> dict:
        try:
            data = yfinance.Ticker(normalized).history(
                period=period,
                interval=interval,
            )
            if data.empty:
                return {"error": f"No intraday data for {normalized}"}
            return {
                "ticker": normalized,
                "interval": interval,
                "bars": [
                    {
                        "time": iso_utc(t),
                        "open": round(float(row["Open"]), 2),
                        "high": round(float(row["High"]), 2),
                        "low": round(float(row["Low"]), 2),
                        "close": round(float(row["Close"]), 2),
                        "volume": int(row["Volume"]),
                    }
                    for t, row in data.iterrows()
                ],
                "predictions": [
                    prediction
                    for prediction in PredictionStore().get_history(
                        ticker=normalized,
                        limit=500,
                    )
                    if prediction["interval"].endswith(("m", "h"))
                ][:100],
            }
        except Exception as exc:
            return {"error": str(exc), "ticker": normalized}

    return await run_in_threadpool(
        _cached_payload,
        f"intraday:{normalized}:{period}:{interval}",
        load,
        ttl_seconds=30,
        fallback={
            "ticker": normalized,
            "interval": interval,
            "bars": [],
            "predictions": [],
            "error": "Intraday data is refreshing",
        },
    )


@app.get("/api/stock/{ticker}/daytrading/predict")
async def api_daytrading_predict(ticker: str):
    """Get day trading prediction using the intraday HorizonModel."""
    normalized = ticker.upper()
    predictions = _stored_predictions([normalized])
    intraday = [
        prediction
        for prediction in predictions
        if prediction["model_name"] in {
            "model_realtime_5m",
            "model_intraday",
        }
    ]
    intraday.sort(
        key=lambda prediction: (
            prediction["model_name"] != "model_realtime_5m",
            prediction["horizon"],
        )
    )
    if intraday:
        prediction = intraday[0]
        prediction["horizon_bars"] = prediction["horizon"]
        prediction["model_status"] = "ready"
        return prediction

    refreshing = _schedule_prediction_refresh(normalized)
    return {
        "ticker": normalized,
        "direction": "N/A",
        "predicted_return": 0,
        "confidence": 0,
        "model_status": "refreshing" if refreshing else "not_trained",
        "message": (
            "Prediction is being generated in the background"
            if refreshing
            else "No intraday model is trained"
        ),
    }


@app.post("/api/daytrading/train")
async def api_daytrading_train(
    tickers: str = Query("AAPL,MSFT,NVDA,TSLA,AMD", description="Comma-separated tickers"),
):
    """Train the intraday day trading model (non-blocking)."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    current = training_tracker.get("intraday")
    if current and current["state"] == "training":
        return {"status": "already_training"}

    def _train():
        try:
            from ml.multi_horizon import HorizonModel, HORIZONS

            config = HORIZONS["intraday"]
            model = HorizonModel(config)
            training_tracker.start("intraday", total_tickers=len(ticker_list))
            logger.info("Training intraday model on %s...", ticker_list)

            model.train(ticker_list)
            model.save()

            acc = model._training_stats.get("cv_accuracy", 0) if model._training_stats else 0
            training_tracker.finish("intraday", accuracy=acc)
            logger.info("Intraday model trained. CV=%.3f", acc)
        except Exception as e:
            training_tracker.fail("intraday", str(e))
            logger.error("Intraday training failed: %s", e)

    thread = threading.Thread(target=_train, daemon=True)
    thread.start()
    return {"status": "started", "tickers": ticker_list}


@app.get("/api/timeframes")
async def api_timeframes():
    """Return available timeframes for strategy backtests."""
    return {
        "timeframes": list(TIMEFRAMES.keys()),
        "default": DEFAULT_TIMEFRAME,
        "labels": {
            "1mo": "1 Month",
            "3mo": "3 Months",
            "6mo": "6 Months",
            "ytd": "Year to Date",
            "1y": "1 Year",
            "2y": "2 Years",
            "5y": "5 Years",
            "max": "Max",
        },
    }


# ── Startup ──

@app.on_event("startup")
async def on_startup():
    """Auto-start model training when the server boots."""
    def prewarm() -> None:
        try:
            _web_cache.get(
                "market-overview",
                lambda: {
                    "indices": {
                        name: {
                            "price": round(float(data["Close"].iloc[-1]), 2),
                            "change": round(
                                float(data["Close"].pct_change().iloc[-1] * 100),
                                2,
                            ),
                        }
                        for symbol, name in [
                            ("SPY", "S&P 500"),
                            ("QQQ", "Nasdaq 100"),
                            ("IWM", "Russell 2000"),
                            ("DIA", "Dow Jones"),
                            ("^VIX", "VIX"),
                        ]
                        if not (
                            data := yfinance.Ticker(symbol).history(period="5d")
                        ).empty
                        and len(data) >= 2
                    }
                },
                ttl_seconds=60,
                wait_seconds=0,
                fallback={"indices": {}},
            )
        except Exception:
            logger.exception("Web cache prewarm failed")

    threading.Thread(target=prewarm, name="web-cache-prewarm", daemon=True).start()
    if os.environ.get("WEBAPP_AUTO_TRAIN", "1").strip().lower() in {
        "0", "false", "no", "off",
    }:
        logger.info("Server starting with web-app auto-training disabled")
        return
    logger.info("Server starting — launching background model training...")
    thread = threading.Thread(target=_auto_train_models, daemon=True)
    thread.start()


def create_app():
    """Create and return the FastAPI app (for programmatic use)."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "webapp.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
