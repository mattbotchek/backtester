#!/usr/bin/env python
"""Single-command startup for the Stock Predictor 24/7 system.

This script starts:
  1. Web server (FastAPI on port 8000)
  2. ML headless prediction daemon
  3. Sentiment data scraper (periodic)
  4. Model retraining scheduler

Usage:
  python run_server.py                    # Start everything
  python run_server.py --port 8080        # Custom port
  python run_server.py --no-scraper       # Skip sentiment scraping
  python run_server.py --retrain-hours 6  # Retrain every 6 hours
"""

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("server")

# Graceful shutdown
shutdown_event = threading.Event()


def signal_handler(signum, frame):
    logger.info("Shutdown signal received...")
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_web_server(port: int):
    """Run the FastAPI web server."""
    import uvicorn
    logger.info("Starting web server on port %d...", port)
    uvicorn.run(
        "webapp.app:app",
        host="0.0.0.0",
        port=port,
        log_level="warning",
    )


def run_sentiment_scraper(interval_minutes: int = 30):
    """Periodically scrape sentiment data."""
    from ml.sentiment.scrapers import scrape_all

    logger.info("Starting sentiment scraper (every %d min)...", interval_minutes)
    while not shutdown_event.is_set():
        try:
            result = scrape_all()
            logger.info(
                "Scraped %d articles, Fear/Greed: %s",
                result.get("total_articles", 0),
                result.get("fear_greed", "N/A"),
            )
        except Exception as e:
            logger.error("Scraper error: %s", e)

        shutdown_event.wait(interval_minutes * 60)


def run_prediction_daemon(interval_minutes: int = 60):
    """Run ML predictions on S&P 500 stocks periodically."""
    logger.info("Starting prediction daemon (every %d min)...", interval_minutes)

    while not shutdown_event.is_set():
        try:
            from ml.model import StockPredictor
            from webapp.app import POPULAR_TICKERS

            predictor = StockPredictor()
            predictor.load()

            up_count, down_count = 0, 0
            for ticker in POPULAR_TICKERS[:20]:
                try:
                    pred = predictor.predict_ticker(ticker)
                    if pred.direction == 1:
                        up_count += 1
                    elif pred.direction == -1:
                        down_count += 1
                except Exception:
                    pass

            logger.info(
                "Predictions: %d UP, %d DOWN, %d total",
                up_count, down_count, up_count + down_count,
            )
        except FileNotFoundError:
            logger.warning("No trained model found. Train first: python main.py --mode ml-train")
        except Exception as e:
            logger.error("Prediction error: %s", e)

        shutdown_event.wait(interval_minutes * 60)


def run_retrainer(interval_hours: int = 24):
    """Periodically retrain the ML model with fresh data."""
    logger.info("Starting retrainer (every %d hours)...", interval_hours)

    # Wait a bit before first retrain
    shutdown_event.wait(300)

    while not shutdown_event.is_set():
        try:
            from ml.model import StockPredictor

            logger.info("Retraining model...")
            tickers = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
                "JNJ", "V", "PG", "UNH", "HD", "DIS", "MA", "NFLX",
                "XOM", "CVX", "WMT", "BA", "GS", "BAC", "PFE", "AMD",
                "ADBE", "CRM", "INTC", "CSCO", "PEP", "KO", "MRK",
            ]

            predictor = StockPredictor()
            stats = predictor.train(tickers, timeline="5y")
            predictor.save()

            logger.info(
                "Retrain complete: CV=%.3f, %d samples",
                stats["cv_accuracy_mean"], stats["n_samples"],
            )
        except Exception as e:
            logger.error("Retrain error: %s", e)

        shutdown_event.wait(interval_hours * 3600)


def main():
    parser = argparse.ArgumentParser(description="Stock Predictor 24/7 Server")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--no-scraper", action="store_true", help="Disable sentiment scraper")
    parser.add_argument("--no-predictions", action="store_true", help="Disable prediction daemon")
    parser.add_argument("--no-retrain", action="store_true", help="Disable auto-retraining")
    parser.add_argument("--scrape-interval", type=int, default=30, help="Scrape interval (minutes)")
    parser.add_argument("--predict-interval", type=int, default=60, help="Prediction interval (minutes)")
    parser.add_argument("--retrain-hours", type=int, default=24, help="Retrain interval (hours)")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("STOCK PREDICTOR SERVER")
    logger.info("  Web UI: http://localhost:%d", args.port)
    logger.info("  Sentiment scraper: %s", "OFF" if args.no_scraper else f"every {args.scrape_interval}min")
    logger.info("  Predictions: %s", "OFF" if args.no_predictions else f"every {args.predict_interval}min")
    logger.info("  Auto-retrain: %s", "OFF" if args.no_retrain else f"every {args.retrain_hours}h")
    logger.info("=" * 50)

    threads = []

    # Start background services
    if not args.no_scraper:
        t = threading.Thread(target=run_sentiment_scraper, args=(args.scrape_interval,), daemon=True)
        t.start()
        threads.append(t)

    if not args.no_predictions:
        t = threading.Thread(target=run_prediction_daemon, args=(args.predict_interval,), daemon=True)
        t.start()
        threads.append(t)

    if not args.no_retrain:
        t = threading.Thread(target=run_retrainer, args=(args.retrain_hours,), daemon=True)
        t.start()
        threads.append(t)

    # Run web server (blocks until shutdown)
    run_web_server(args.port)


if __name__ == "__main__":
    main()
