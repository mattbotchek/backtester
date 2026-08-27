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

import logging
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies import get_registry

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

# Cache for predictions
_prediction_cache: dict = {}
_cache_timestamp: Optional[datetime] = None


# ── Helper functions ──

def _get_stock_data(ticker: str, period: str = "1y") -> dict:
    """Get stock data from yfinance."""
    try:
        stock = yfinance.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info if hasattr(stock, "info") else {}

        if hist.empty:
            return {"error": f"No data for {ticker}"}

        return {
            "ticker": ticker,
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
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
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


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

        predictor = StockPredictor()
        predictor.load()
        pred = predictor.predict(ticker)
        return {
            "direction": pred.direction,
            "predicted_return": round(pred.predicted_return_pct, 2),
            "confidence": round(pred.confidence * 100, 1),
            "outliers": pred.outliers,
        }
    except Exception as e:
        logger.debug("Prediction failed for %s: %s", ticker, e)
        return {
            "direction": "N/A",
            "predicted_return": 0,
            "confidence": 0,
            "outliers": [],
        }


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
            pass
    return {"popular": results}


@app.get("/api/stock/{ticker}")
async def api_stock(ticker: str):
    """Get stock data and quick stats."""
    return JSONResponse(_get_stock_data(ticker.upper()))


@app.get("/api/stock/{ticker}/strategies")
async def api_strategies(
    ticker: str,
    timeline: str = Query(DEFAULT_TIMEFRAME, description="Backtest timeframe"),
):
    """Run all strategies on a stock with a given timeframe."""
    results = _run_strategies(ticker.upper(), timeline)
    return {"ticker": ticker.upper(), "timeline": timeline, "strategies": results}


@app.get("/api/stock/{ticker}/prediction")
async def api_prediction(ticker: str):
    """Get ML prediction for a stock."""
    pred = _get_ml_prediction(ticker.upper())
    return {"ticker": ticker.upper(), "prediction": pred}


@app.get("/api/market/overview")
async def api_market_overview():
    """Get market overview (major indices + top movers)."""
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
            pass
    return {"indices": indices}


@app.get("/api/market/predictions")
async def api_market_predictions():
    """Get market-wide predictions: overall direction and top movers."""
    predictions = []
    for ticker in POPULAR_TICKERS:
        pred = _get_ml_prediction(ticker)
        if pred["direction"] != "N/A":
            predictions.append({"ticker": ticker, **pred})

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
    global _prediction_cache, _cache_timestamp

    if (
        _cache_timestamp
        and (datetime.now() - _cache_timestamp) < timedelta(hours=1)
    ):
        return {"predictions": _prediction_cache, "cached": True}

    predictions = {}
    for ticker in POPULAR_TICKERS:
        pred = _get_ml_prediction(ticker)
        predictions[ticker] = pred

    _prediction_cache = predictions
    _cache_timestamp = datetime.now()
    return {"predictions": predictions, "cached": False}


@app.get("/api/training/status")
async def api_training_status():
    """Get current training progress for all models."""
    return {"models": training_tracker.get_all()}


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
        from ml.sentiment.scrapers import scrape_all, get_aggregated_sentiment

        scrape_all()
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
    try:
        data = yfinance.Ticker(ticker.upper()).history(
            period=period, interval=interval,
        )
        if data.empty:
            return {"error": f"No intraday data for {ticker}"}

        return {
            "ticker": ticker.upper(),
            "interval": interval,
            "bars": [
                {
                    "time": t.strftime("%Y-%m-%d %H:%M"),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
                for t, row in data.iterrows()
            ],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/stock/{ticker}/daytrading/predict")
async def api_daytrading_predict(ticker: str):
    """Get day trading prediction using the intraday HorizonModel."""
    try:
        import numpy as np
        from ml.multi_horizon import HorizonModel, HORIZONS
        from ml.features import compute_intraday_features

        config = HORIZONS["intraday"]
        model = HorizonModel(config)
        model.load()

        data = yfinance.Ticker(ticker.upper()).history(
            period=config.data_period, interval=config.data_interval,
        )
        if data.empty:
            return {"error": f"No intraday data for {ticker}"}

        features = compute_intraday_features(data)
        if features.empty:
            return {"error": "Insufficient intraday data for features"}

        # Use last bar
        available = [f for f in model.feature_names if f in features.columns]
        missing = [f for f in model.feature_names if f not in features.columns]
        if missing:
            # Fill missing features with 0
            for col in missing:
                features[col] = 0.0

        X = features.iloc[[-1]][model.feature_names]
        X_scaled = model.scaler.transform(X)
        direction, pred_return, confidence = model.predict(X_scaled)

        return {
            "ticker": ticker.upper(),
            "direction": "UP" if pred_return >= 0 else "DOWN",
            "predicted_return": round(pred_return * 100, 3),
            "confidence": round(confidence * 100, 1),
            "horizon_bars": config.horizon,
            "interval": config.data_interval,
            "model_status": "ready",
        }
    except FileNotFoundError:
        return {
            "ticker": ticker.upper(),
            "direction": "N/A",
            "predicted_return": 0,
            "confidence": 0,
            "model_status": "not_trained",
            "message": "Intraday model not trained. Use: python main.py --mode ml-horizons-train --horizon intraday",
        }
    except Exception as e:
        logger.debug("Day trading prediction failed for %s: %s", ticker, e)
        return {
            "ticker": ticker.upper(),
            "direction": "N/A",
            "predicted_return": 0,
            "confidence": 0,
            "model_status": "error",
            "message": str(e),
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
