"""Stock Market Backtester & ML Predictor.

Main entry point supporting multiple modes:
  - Interactive menu (default)
  - CLI arguments for scripted usage
  - Headless/continuous analysis mode

Usage:
    python main.py                          # Interactive menu
    python main.py --mode backtest --tickers AAPL MSFT --strategy BuyAndHold
    python main.py --mode compare --tickers AAPL --strategy all
    python main.py --mode headless --interval 60
    python main.py --mode ml-train --tickers AAPL MSFT GOOGL
    python main.py --mode ml-predict --tickers TSLA
    python main.py --mode ml-backtest --tickers AAPL MSFT
    python main.py --mode ml-headless --interval 3600
"""

import argparse
import logging
import sys
import time
from datetime import datetime

from strategies import get_registry, list_strategies, TradeResult
from backtesting.engine import (
    BacktestConfig,
    BacktestReport,
    run_backtest,
    print_report,
    check_ticker_exists,
)

logger = logging.getLogger(__name__)

# Broad default tickers across sectors
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "JPM", "JNJ",
    "KO", "XOM", "BA", "SPY", "QQQ",
]

# Training tickers — broad cross-industry coverage for ML
ML_TRAINING_TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD", "INTC", "CRM",
    "ORCL", "ADBE", "CSCO", "AVGO", "TXN",
    # Finance
    "JPM", "BAC", "GS", "V", "MA", "BRK-B", "WFC", "C", "AXP", "BLK",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
    # Consumer
    "KO", "PEP", "PG", "WMT", "COST", "MCD", "NKE", "SBUX", "HD", "LOW",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Industrials
    "BA", "CAT", "UNP", "HON", "GE", "MMM", "RTX",
    # Real Estate / Utilities / Materials
    "AMT", "PLD", "NEE", "DUK", "LIN", "APD",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLV",
]


# ── Interactive Menu ──────────────────────────────────────────────────

def _print_strategies():
    """Display all available strategies with descriptions."""
    registry = get_registry()
    print("\nAvailable Strategies:")
    print("-" * 60)
    for i, name in enumerate(sorted(registry.keys())):
        cls = registry[name]
        novel = " [NOVEL]" if getattr(cls, "is_novel", False) else ""
        desc = getattr(cls, "description", "")
        print(f"  {i:2d}) {name}{novel}")
        if desc:
            print(f"      {desc}")
    print()


def _choose_strategy() -> str:
    """Prompt user to choose a strategy by number or name."""
    _print_strategies()
    names = sorted(get_registry().keys())
    choice = input("Enter strategy number or name: ").strip()
    try:
        idx = int(choice)
        return names[idx]
    except (ValueError, IndexError):
        if choice in get_registry():
            return choice
        print(f"Invalid choice: {choice}")
        return _choose_strategy()


def _get_user_params() -> dict:
    """Collect common backtest parameters from user."""
    timeline = input("Timeline (e.g. 1y, 6mo, ytd) [1y]: ").strip() or "1y"
    interval = input("Interval (e.g. 1d, 1wk) [1d]: ").strip() or "1d"
    cash = input("Starting cash [10000]: ").strip() or "10000"
    verbose = input("Verbose output? (y/n) [n]: ").strip().lower() == "y"
    return {
        "timeline": timeline,
        "interval": interval,
        "cash": float(cash),
        "verbose": verbose,
    }


def interactive_single_strategy():
    """Run a single strategy on a single ticker."""
    strategy_name = _choose_strategy()
    ticker = input("Enter ticker: ").strip().upper()
    params = _get_user_params()

    config = BacktestConfig(
        tickers=[ticker],
        strategy_names=[strategy_name],
        timeline=params["timeline"],
        interval=params["interval"],
        starting_cash=params["cash"],
        verbose=params["verbose"],
    )
    report = run_backtest(config)
    print_report(report)


def interactive_compare_all():
    """Compare all strategies on a single ticker."""
    ticker = input("Enter ticker: ").strip().upper()
    params = _get_user_params()

    config = BacktestConfig(
        tickers=[ticker],
        strategy_names=[],  # empty = all
        timeline=params["timeline"],
        interval=params["interval"],
        starting_cash=params["cash"],
        verbose=params["verbose"],
    )
    report = run_backtest(config)
    print_report(report)


def interactive_multi_ticker():
    """Run a strategy across multiple tickers."""
    strategy_name = _choose_strategy()
    tickers_input = input("Enter tickers (comma-separated): ").strip().upper()
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    params = _get_user_params()

    config = BacktestConfig(
        tickers=tickers,
        strategy_names=[strategy_name],
        timeline=params["timeline"],
        interval=params["interval"],
        starting_cash=params["cash"],
        verbose=params["verbose"],
    )
    report = run_backtest(config)
    print_report(report)


def interactive_all_tickers():
    """Run a strategy on all tickers from allsymbols.txt."""
    strategy_name = _choose_strategy()
    params = _get_user_params()

    try:
        with open("allsymbols.txt") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    except FileNotFoundError:
        print("allsymbols.txt not found. Enter tickers manually.")
        return interactive_multi_ticker()

    config = BacktestConfig(
        tickers=tickers,
        strategy_names=[strategy_name],
        timeline=params["timeline"],
        interval=params["interval"],
        starting_cash=params["cash"],
        verbose=params["verbose"],
    )
    report = run_backtest(config)
    print_report(report)

    # Save to CSV
    filename = f"results/backtest_{strategy_name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    report.to_csv(filename)
    print(f"Results saved to {filename}")


def interactive_compare_strategies_on_ticker():
    """Compare selected strategies on a single ticker."""
    strategies = []
    while True:
        strategies.append(_choose_strategy())
        more = input("Add another strategy? (y/n) [n]: ").strip().lower()
        if more != "y":
            break

    ticker = input("Enter ticker: ").strip().upper()
    params = _get_user_params()

    config = BacktestConfig(
        tickers=[ticker],
        strategy_names=strategies,
        timeline=params["timeline"],
        interval=params["interval"],
        starting_cash=params["cash"],
        verbose=params["verbose"],
    )
    report = run_backtest(config)
    print_report(report)


def interactive_menu():
    """Main interactive menu loop."""
    modes = [
        ("Run single strategy on a ticker", interactive_single_strategy),
        ("Compare ALL strategies on a ticker", interactive_compare_all),
        ("Run strategy on multiple tickers", interactive_multi_ticker),
        ("Run strategy on all tickers (allsymbols.txt)", interactive_all_tickers),
        ("Compare selected strategies on a ticker", interactive_compare_strategies_on_ticker),
        ("List all strategies", lambda: _print_strategies()),
    ]

    while True:
        print("\n" + "=" * 50)
        print("  STOCK BACKTESTER")
        print("=" * 50)
        for i, (label, _) in enumerate(modes):
            print(f"  {i}) {label}")
        print(f"  q) Quit")
        print()

        choice = input("Choice: ").strip().lower()
        if choice == "q":
            break
        try:
            idx = int(choice)
            modes[idx][1]()
        except (ValueError, IndexError):
            print("Invalid choice.")


# ── Headless Strategy Mode ────────────────────────────────────────────

def headless_mode(
    tickers: list[str],
    strategy_names: list[str],
    timeline: str,
    interval: str,
    cash: float,
    loop_interval: int,
):
    """Run continuous strategy analysis in a loop."""
    import signal

    running = True
    def handle_signal(sig, frame):
        nonlocal running
        print("\nShutting down headless mode...")
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cycle = 0
    while running:
        cycle += 1
        print(f"\n{'=' * 70}")
        print(f"  HEADLESS CYCLE #{cycle} — {datetime.now().isoformat()}")
        print(f"{'=' * 70}")

        config = BacktestConfig(
            tickers=tickers,
            strategy_names=strategy_names,
            timeline=timeline,
            interval=interval,
            starting_cash=cash,
        )
        report = run_backtest(config)
        print_report(report)

        filename = f"results/headless_{datetime.now():%Y%m%d_%H%M%S}.csv"
        report.to_csv(filename)

        if not running:
            break
        print(f"  Next cycle in {loop_interval}s... (Ctrl+C to stop)")
        for _ in range(loop_interval):
            if not running:
                break
            time.sleep(1)


# ── CLI Entry Point ───────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stock Market Backtester & ML Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive menu
  python main.py --mode backtest -t AAPL -s BuyAndHold
  python main.py --mode compare -t AAPL             # All strategies
  python main.py --mode headless -t AAPL MSFT --interval 300
  python main.py --mode ml-train                    # Train ML model
  python main.py --mode ml-predict -t TSLA           # Predict one ticker
  python main.py --mode ml-backtest                 # Walk-forward test
  python main.py --mode ml-headless --interval 3600 # 24/7 ML analysis
  python main.py --mode list                        # List strategies
        """,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=[
            "interactive", "backtest", "compare", "headless",
            "ml-train", "ml-predict", "ml-backtest", "ml-headless",
            "ml-horizons-train", "ml-horizons-backtest",
            "list",
        ],
        default="interactive",
        help="Operating mode (default: interactive)",
    )
    parser.add_argument("--tickers", "-t", nargs="+", default=None, help="Ticker symbols")
    parser.add_argument("--strategy", "-s", nargs="+", default=None, help="Strategy names")
    parser.add_argument("--timeline", default="1y", help="Data timeline (default: 1y)")
    parser.add_argument("--interval", type=int, default=3600, help="Headless loop interval in seconds")
    parser.add_argument("--data-interval", default="1d", help="OHLCV data interval (default: 1d)")
    parser.add_argument("--cash", type=float, default=10000, help="Starting cash (default: 10000)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", default=None, help="Output CSV file")
    parser.add_argument("--model", default="stock_predictor", help="ML model name")
    parser.add_argument(
        "--horizon", nargs="+", default=None,
        help="Horizon names for multi-horizon mode (intraday, short_1d, short_5d, medium_20d, long_60d, long_252d)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tickers = args.tickers or DEFAULT_TICKERS
    strategy_names = args.strategy or []

    match args.mode:
        case "interactive":
            interactive_menu()

        case "list":
            _print_strategies()

        case "backtest":
            config = BacktestConfig(
                tickers=tickers,
                strategy_names=strategy_names,
                timeline=args.timeline,
                interval=args.data_interval,
                starting_cash=args.cash,
                verbose=args.verbose,
            )
            report = run_backtest(config)
            print_report(report)
            if args.output:
                report.to_csv(args.output)

        case "compare":
            config = BacktestConfig(
                tickers=tickers,
                strategy_names=[],  # all strategies
                timeline=args.timeline,
                interval=args.data_interval,
                starting_cash=args.cash,
                verbose=args.verbose,
            )
            report = run_backtest(config)
            print_report(report)
            if args.output:
                report.to_csv(args.output)

        case "headless":
            headless_mode(
                tickers=tickers,
                strategy_names=strategy_names,
                timeline=args.timeline,
                interval=args.data_interval,
                cash=args.cash,
                loop_interval=args.interval,
            )

        case "ml-train":
            from ml.model import StockPredictor
            training_tickers = args.tickers or ML_TRAINING_TICKERS
            predictor = StockPredictor()
            stats = predictor.train(training_tickers, timeline=args.timeline)
            predictor.save(args.model)
            print(f"\nTraining complete:")
            print(f"  Samples: {stats['n_samples']}")
            print(f"  Features: {stats['n_features']}")
            print(f"  CV Accuracy: {stats['cv_accuracy_mean']:.3f} ± {stats['cv_accuracy_std']:.3f}")

        case "ml-predict":
            from ml.model import StockPredictor
            predictor = StockPredictor()
            predictor.load(args.model)
            for ticker in tickers:
                try:
                    pred = predictor.predict(ticker, timeline=args.timeline)
                    print(f"  {pred}")
                except Exception as e:
                    print(f"  {ticker}: Error — {e}")

        case "ml-backtest":
            from ml.backtest import walk_forward_backtest
            test_tickers = args.tickers or ML_TRAINING_TICKERS[:15]
            result = walk_forward_backtest(
                tickers=ML_TRAINING_TICKERS,  # Train on full pool
                test_tickers=test_tickers,     # Test on subset
            )
            print(result.summary())

        case "ml-headless":
            from ml.headless import run_headless
            run_headless(
                model_name=args.model,
                tickers=args.tickers,
                interval_seconds=args.interval,
                output_file=args.output,
            )

        case "ml-horizons-train":
            from ml.multi_horizon import train_all_horizons
            horizons = args.horizon
            results = train_all_horizons(ML_TRAINING_TICKERS, horizons=horizons)
            for name, stats in results.items():
                if "error" in stats:
                    print(f"  {name}: FAILED - {stats['error']}")
                else:
                    print(f"  {name}: CV {stats['cv_accuracy_mean']:.3f} +/- {stats['cv_accuracy_std']:.3f}  ({stats['n_samples']} samples)")

        case "ml-horizons-backtest":
            from ml.multi_horizon import walk_forward_horizon, HORIZONS
            horizons = args.horizon or list(HORIZONS.keys())
            for name in horizons:
                if name == "intraday":
                    print(f"  {name}: skipped (intraday walk-forward not supported)")
                    continue
                print(f"--- {name} ---")
                result = walk_forward_horizon(name, ML_TRAINING_TICKERS)
                if "error" in result:
                    print(f"  ERROR: {result['error']}")
                else:
                    print(f"  Accuracy: {result['accuracy']:.3f}")
                    print(f"  Directional Accuracy: {result['directional_accuracy']:.3f}")
                    print(f"  Predictions: {result['total_predictions']}")
                    print(result['classification_report'])


if __name__ == "__main__":
    main()