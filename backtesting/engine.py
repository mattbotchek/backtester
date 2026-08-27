"""Backtesting engine — runs strategies across tickers and collects metrics.

Supports:
  - Single strategy / single ticker
  - Single strategy / multiple tickers
  - All strategies / single ticker
  - All strategies / all tickers
  - Comparative analysis with ranked output
"""

import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yfinance

from strategies import Strategy, TradeResult, get_registry

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    tickers: list[str]
    strategy_names: list[str]  # empty = all
    timeline: str = "1y"
    interval: str = "1d"
    starting_cash: float = 10000.0
    verbose: bool = False
    output_dir: str = "results"
    max_workers: int = 4


@dataclass
class BacktestReport:
    """Aggregated results from a backtest run."""

    results: list[TradeResult] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def best_result(self) -> Optional[TradeResult]:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.roi_percent)

    @property
    def worst_result(self) -> Optional[TradeResult]:
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.roi_percent)

    def summary_table(self) -> list[dict]:
        """Return sorted summary for display."""
        return sorted(
            [r.to_dict() for r in self.results],
            key=lambda x: x["ROI"],
            reverse=True,
        )

    def to_csv(self, path: str):
        """Write results to CSV."""
        if not self.results:
            return
        rows = self.summary_table()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Results written to %s", path)


def check_ticker_exists(ticker: str) -> bool:
    """Verify a ticker is valid by checking for recent data."""
    yf_logger = logging.getLogger("yfinance")
    was_disabled = yf_logger.disabled
    yf_logger.disabled = True
    try:
        history = yfinance.Ticker(ticker).history(period="5d")
        return not history.empty
    except Exception:
        return False
    finally:
        yf_logger.disabled = was_disabled


def _run_single(
    strategy_name: str,
    strategy_cls: type,
    ticker: str,
    config: BacktestConfig,
) -> Optional[TradeResult]:
    """Run one strategy on one ticker. Returns None on error."""
    try:
        strat = strategy_cls(
            config.timeline, config.interval, ticker, config.starting_cash
        )
        return strat.calculate(verbose=config.verbose)
    except Exception as e:
        logger.warning("Error running %s on %s: %s", strategy_name, ticker, e)
        return None


def run_backtest(config: BacktestConfig) -> BacktestReport:
    """Execute a full backtest run across tickers and strategies."""
    registry = get_registry()
    report = BacktestReport()

    # Determine strategies to run
    if config.strategy_names:
        strategies = {
            name: registry[name]
            for name in config.strategy_names
            if name in registry
        }
    else:
        strategies = registry

    if not strategies:
        logger.error("No valid strategies selected.")
        return report

    # Validate tickers
    valid_tickers = []
    for ticker in config.tickers:
        if check_ticker_exists(ticker):
            valid_tickers.append(ticker)
        else:
            report.errors.append({"ticker": ticker, "error": "Ticker not found"})
            logger.warning("Ticker %s not found, skipping.", ticker)

    if not valid_tickers:
        logger.error("No valid tickers to test.")
        return report

    # Run backtests
    tasks = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        for ticker in valid_tickers:
            for name, cls in strategies.items():
                future = executor.submit(_run_single, name, cls, ticker, config)
                tasks.append((future, name, ticker))

        for future, name, ticker in tasks:
            try:
                result = future.result(timeout=120)
                if result:
                    report.results.append(result)
                else:
                    report.errors.append(
                        {"ticker": ticker, "strategy": name, "error": "Returned None"}
                    )
            except Exception as e:
                report.errors.append(
                    {"ticker": ticker, "strategy": name, "error": str(e)}
                )

    return report


def print_report(report: BacktestReport):
    """Pretty-print a backtest report to the console."""
    if not report.results:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print(f"  BACKTEST REPORT — {report.timestamp}")
    print("=" * 90)
    print(
        f"{'Strategy':<30} {'Ticker':<8} {'ROI%':>10} {'Win Rate':>10} "
        f"{'Max DD':>10} {'Sharpe':>10}"
    )
    print("-" * 90)

    for row in report.summary_table():
        print(
            f"{row['Strategy']:<30} {row['Ticker']:<8} {row['ROI']:>10.2f}% "
            f"{row['Win Rate']:>10} {row['Max Drawdown']:>10} {row['Sharpe Ratio']:>10}"
        )

    print("-" * 90)
    best = report.best_result
    worst = report.worst_result
    if best:
        print(f"  Best:  {best.strategy_name} on {best.ticker} — {best.roi_percent:.2f}%")
    if worst:
        print(f"  Worst: {worst.strategy_name} on {worst.ticker} — {worst.roi_percent:.2f}%")
    print(f"  Errors: {len(report.errors)}")
    print("=" * 90 + "\n")
