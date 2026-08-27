"""Stock Predictor Web Application.

FastAPI-based web server providing:
  - Dashboard with S&P 500 predictions
  - Per-stock analysis with strategies and ML predictions
  - Real-time charts with prediction overlays
  - 24/7 training pipeline management
  - Day trading minute-by-minute predictions

Run with: python -m webapp.app
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies import get_registry
from backtesting.engine import BacktestConfig, run_backtest

logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Predictor", version="1.0")

WEBAPP_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

# S&P 500 tickers (representative subset — full list would be 500+)
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL",
    "NFLX", "ADBE", "CRM", "INTC", "CSCO", "PEP", "KO", "MRK",
    "ABT", "XOM", "CVX", "WMT", "COST", "TGT", "LOW", "NKE",
    "SBUX", "BA", "CAT", "DE", "GE", "MMM", "HON", "UPS",
    "GS", "MS", "C", "BAC", "WFC", "AXP", "LLY", "PFE",
    "ABBV", "BMY", "GILD", "TMO", "DHR", "NEE", "DUK", "SO",
    "AEP", "SRE", "AMT", "PLD", "SPG", "O", "AMD", "QCOM",
    "TXN", "AVGO", "MU", "AMAT", "COP", "EOG", "SLB", "PSX",
    "MCO", "SPGI", "CME", "ICE", "SCHW", "BLK", "T", "VZ",
    "TMUS", "CMCSA", "CHTR", "FDX", "UNP", "CSX", "NSC",
    "LIN", "APD", "SHW", "ECL", "DD", "EMR", "ETN", "ROK",
    "ISRG", "SYK", "MDT", "ZTS", "DXCM", "IDXX",
]

# Cache for predictions
_prediction_cache: dict = {}
_cache_timestamp: Optional[datetime] = None


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
            "change_5d": float(hist["Close"].pct_change(5).iloc[-1] * 100) if len(hist) > 5 else 0,
            "change_20d": float(hist["Close"].pct_change(20).iloc[-1] * 100) if len(hist) > 20 else 0,
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


def _run_strategies(ticker: str) -> list[dict]:
    """Run all strategies on a ticker."""
    results = []
    registry = get_registry()

    for name, strategy_cls in registry.items():
        try:
            strategy = strategy_cls(
                timeline="1y", interval="1d", ticker=ticker, cash=10000,
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
        pred = predictor.predict_ticker(ticker)
        return {
            "direction": "UP" if pred.direction == 1 else ("DOWN" if pred.direction == -1 else "FLAT"),
            "predicted_return": round(pred.predicted_return * 100, 2),
            "confidence": round(pred.confidence * 100, 1),
            "outliers": pred.outliers,
        }
    except Exception:
        return {"direction": "N/A", "predicted_return": 0, "confidence": 0, "outliers": []}


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard — overview of market and top predictions."""
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_detail(request: Request, ticker: str):
    """Per-stock detail page with strategies and predictions."""
    return templates.TemplateResponse(request, "stock_detail.html", {"ticker": ticker.upper()})


@app.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request):
    """Strategies overview page."""
    return templates.TemplateResponse(request, "strategies.html")


@app.get("/ml", response_class=HTMLResponse)
async def ml_page(request: Request):
    """ML model control and status page."""
    return templates.TemplateResponse(request, "ml_control.html")


# ── API Endpoints ──

@app.get("/api/tickers")
async def api_tickers(q: str = Query("", description="Search query")):
    """Search/list available tickers."""
    if q:
        matches = [t for t in SP500_TICKERS if q.upper() in t]
        return {"tickers": matches[:20]}
    return {"tickers": SP500_TICKERS}


@app.get("/api/stock/{ticker}")
async def api_stock(ticker: str):
    """Get stock data and quick stats."""
    return JSONResponse(_get_stock_data(ticker.upper()))


@app.get("/api/stock/{ticker}/strategies")
async def api_strategies(ticker: str):
    """Run all strategies on a stock."""
    results = _run_strategies(ticker.upper())
    return {"ticker": ticker.upper(), "strategies": results}


@app.get("/api/stock/{ticker}/prediction")
async def api_prediction(ticker: str):
    """Get ML prediction for a stock."""
    pred = _get_ml_prediction(ticker.upper())
    return {"ticker": ticker.upper(), "prediction": pred}


@app.get("/api/market/overview")
async def api_market_overview():
    """Get market overview (major indices + top movers)."""
    indices = {}
    for symbol, name in [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("IWM", "Russell 2000"), ("^VIX", "VIX")]:
        try:
            data = yfinance.Ticker(symbol).history(period="2d")
            if not data.empty and len(data) >= 2:
                indices[name] = {
                    "price": round(float(data["Close"].iloc[-1]), 2),
                    "change": round(float(data["Close"].pct_change().iloc[-1] * 100), 2),
                }
        except Exception:
            pass
    return {"indices": indices}


@app.get("/api/predictions/daily")
async def api_daily_predictions():
    """Get daily predictions for all S&P 500 stocks."""
    global _prediction_cache, _cache_timestamp

    # Use cache if < 1 hour old
    if _cache_timestamp and (datetime.now() - _cache_timestamp) < timedelta(hours=1):
        return {"predictions": _prediction_cache, "cached": True}

    predictions = {}
    for ticker in SP500_TICKERS[:20]:  # Limit for speed
        pred = _get_ml_prediction(ticker)
        predictions[ticker] = pred

    _prediction_cache = predictions
    _cache_timestamp = datetime.now()
    return {"predictions": predictions, "cached": False}


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
