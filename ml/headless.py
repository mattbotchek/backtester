"""ML Headless Runner — continuous stock analysis daemon.

Runs the trained ML model in a loop, analyzing stocks on a schedule.
Outputs predictions to console and/or CSV file.

Usage:
    python -m ml.headless --model stock_predictor --interval 3600 --tickers AAPL MSFT GOOGL
"""

import argparse
import csv
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from .model import StockPredictor, Prediction
from .prediction_store import PredictionStore

logger = logging.getLogger(__name__)

# Broad ticker set covering major sectors
DEFAULT_TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD", "INTC", "CRM",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "WFC", "C", "AXP",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
    # Consumer
    "KO", "PEP", "PG", "WMT", "COST", "MCD", "NKE", "SBUX", "HD", "LOW",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrials
    "BA", "CAT", "UNP", "HON", "GE",
    # Real Estate
    "AMT", "PLD", "SPG",
    # Utilities
    "NEE", "DUK", "SO",
    # ETFs for broad market
    "SPY", "QQQ", "IWM", "DIA",
]


def run_headless(
    model_name: str = "stock_predictor",
    tickers: list[str] | None = None,
    interval_seconds: int = 3600,
    output_file: str | None = None,
):
    """Run continuous stock analysis.

    Args:
        model_name: Name of the saved model to load.
        tickers: List of tickers to analyze each cycle.
        interval_seconds: Seconds between analysis cycles.
        output_file: Optional CSV file to append predictions.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    predictor = StockPredictor()
    try:
        predictor.load(model_name)
    except FileNotFoundError:
        logger.error(
            "No trained model found. Train first with:\n"
            "  python main.py --mode ml-train"
        )
        sys.exit(1)

    # CSV setup
    csv_writer = None
    csv_file = None
    if output_file:
        csv_file = open(output_file, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if Path(output_file).stat().st_size == 0:
            csv_writer.writerow([
                "Recorded At", "Model", "Ticker", "Input At", "Target At",
                "Base Price", "Predicted Price", "Direction",
                "Predicted Return %", "Confidence", "Forward Accuracy",
                "Settled Predictions", "Last Trained At", "Outliers",
            ])

    # Graceful shutdown
    running = True
    def handle_signal(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received. Finishing current cycle...")
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cycle = 0
    while running:
        cycle += 1
        timestamp = datetime.now().isoformat()
        print(f"\n{'=' * 70}")
        print(f"  ANALYSIS CYCLE #{cycle} — {timestamp}")
        print(f"{'=' * 70}")
        settlement = PredictionStore().settle_due()
        if settlement["settled"]:
            print(f"  Settled {settlement['settled']} due prediction(s)")

        predictions = []
        for ticker in tickers:
            try:
                pred = predictor.predict(ticker)
                predictions.append(pred)
                print(f"  {pred}")

                if csv_writer:
                    csv_writer.writerow([
                        pred.timestamp, pred.model_name, pred.ticker,
                        pred.input_at, pred.predicted_for,
                        f"{pred.base_price:.4f}",
                        f"{pred.predicted_price:.4f}",
                        pred.direction,
                        f"{pred.predicted_return_pct:.4f}",
                        f"{pred.confidence:.4f}",
                        (
                            f"{pred.forward_accuracy:.4f}"
                            if pred.forward_accuracy is not None
                            else ""
                        ),
                        pred.settled_predictions,
                        pred.last_trained_at or "",
                        "; ".join(pred.outliers) if pred.outliers else "",
                    ])
            except Exception as e:
                logger.warning("  Error predicting %s: %s", ticker, e)

        # Summary
        up = [p for p in predictions if p.direction == "UP"]
        down = [p for p in predictions if p.direction == "DOWN"]
        high_conf = [p for p in predictions if p.confidence > 0.7]

        print(f"\n  Summary: {len(up)} UP, {len(down)} DOWN, "
              f"{len(predictions) - len(up) - len(down)} FLAT")
        if high_conf:
            print(f"  High confidence calls ({len(high_conf)}):")
            for p in sorted(high_conf, key=lambda x: abs(x.predicted_return_pct), reverse=True)[:5]:
                print(f"    {p}")

        if csv_file:
            csv_file.flush()

        if not running:
            break

        print(f"\n  Next cycle in {interval_seconds}s...")
        # Sleep in small chunks so we can respond to signals
        for _ in range(interval_seconds):
            if not running:
                break
            time.sleep(1)

    if csv_file:
        csv_file.close()
    print("\nHeadless runner stopped.")


def main():
    parser = argparse.ArgumentParser(description="ML Stock Predictor — Headless Mode")
    parser.add_argument("--model", default="stock_predictor", help="Saved model name")
    parser.add_argument("--tickers", nargs="+", default=None, help="Tickers to analyze")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between cycles")
    parser.add_argument("--output", default=None, help="CSV output file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_headless(
        model_name=args.model,
        tickers=args.tickers,
        interval_seconds=args.interval,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
