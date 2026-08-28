import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from backtesting.engine import (
    BacktestConfig,
    BacktestReport,
    TradeResult,
    _run_single,
    check_ticker_exists,
    print_report,
    run_backtest,
)


class GoodStrategy:
    name = 'Good'

    def __init__(self, timeline, interval, ticker, cash):
        self.timeline = timeline
        self.interval = interval
        self.ticker = ticker
        self.cash = cash

    def calculate(self, verbose=False):
        return TradeResult(
            ticker=self.ticker,
            strategy_name=self.name,
            initial_cash=10_000.0,
            final_cash=10_500.0,
            roi_percent=5.0,
            total_trades=1,
            winning_trades=1,
        )


class BrokenStrategy(GoodStrategy):
    name = 'Broken'

    def calculate(self, verbose=False):
        raise RuntimeError('boom')


class FakeFuture:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def result(self, timeout=None):
        if self.error:
            raise self.error
        return self.value


class FakeExecutor:
    def __init__(self, futures):
        self._futures = list(futures)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, *args, **kwargs):
        return self._futures.pop(0)


class FakeYFTicker:
    def __init__(self, history):
        self._history = history

    def history(self, **kwargs):
        return self._history


def test_check_ticker_exists_handles_valid_invalid_and_exception():
    good_history = pd.DataFrame({'Close': [1.0]})
    with patch('yfinance.Ticker', return_value=FakeYFTicker(good_history)):
        assert check_ticker_exists('AAPL') is True
    with patch('yfinance.Ticker', return_value=FakeYFTicker(pd.DataFrame())):
        assert check_ticker_exists('BAD') is False
    with patch('yfinance.Ticker', side_effect=RuntimeError('fail')):
        assert check_ticker_exists('ERR') is False


def test_run_single_returns_none_on_strategy_failure():
    config = BacktestConfig(tickers=['AAPL'], strategy_names=['Broken'])
    assert _run_single('Broken', BrokenStrategy, 'AAPL', config) is None


def test_backtest_report_summary_and_csv(tmp_path):
    result_a = TradeResult('AAPL', 'Alpha', 100, 110, 10.0, total_trades=1, winning_trades=1)
    result_b = TradeResult('MSFT', 'Beta', 100, 90, -10.0, total_trades=1, losing_trades=1)
    report = BacktestReport(results=[result_b, result_a])
    rows = report.summary_table()
    assert rows[0]['Strategy'] == 'Alpha'
    assert report.best_result.strategy_name == 'Alpha'
    assert report.worst_result.strategy_name == 'Beta'

    out_path = tmp_path / 'report.csv'
    report.to_csv(str(out_path))
    assert out_path.exists()
    assert 'Alpha' in out_path.read_text(encoding='utf-8')


def test_run_backtest_handles_missing_strategies_and_invalid_tickers(capsys):
    with patch('backtesting.engine.get_registry', return_value={}):
        empty = run_backtest(BacktestConfig(tickers=['BAD'], strategy_names=['Missing']))
    assert empty.results == []

    with (
        patch('backtesting.engine.get_registry', return_value={'Good': GoodStrategy}),
        patch('backtesting.engine.check_ticker_exists', return_value=False),
    ):
        no_tickers = run_backtest(BacktestConfig(tickers=['BAD'], strategy_names=['Good']))
    assert no_tickers.errors[0]['error'] == 'Ticker not found'

    print_report(BacktestReport())
    assert 'No results to display.' in capsys.readouterr().out


def test_run_backtest_collects_returned_none_and_timeout_errors():
    config = BacktestConfig(tickers=['AAPL'], strategy_names=['Good'], max_workers=1)
    with (
        patch('backtesting.engine.get_registry', return_value={'Good': GoodStrategy}),
        patch('backtesting.engine.check_ticker_exists', return_value=True),
        patch('backtesting.engine.ThreadPoolExecutor', return_value=FakeExecutor([FakeFuture(None)])),
    ):
        report = run_backtest(config)
    assert report.errors[0]['error'] == 'Returned None'

    timeout_executor = FakeExecutor([FakeFuture(error=TimeoutError('timed out'))])
    with (
        patch('backtesting.engine.get_registry', return_value={'Good': GoodStrategy}),
        patch('backtesting.engine.check_ticker_exists', return_value=True),
        patch('backtesting.engine.ThreadPoolExecutor', return_value=timeout_executor),
    ):
        timed_out = run_backtest(config)
    assert 'timed out' in timed_out.errors[0]['error']


def test_run_backtest_success_path_and_report_printing():
    config = BacktestConfig(tickers=['AAPL'], strategy_names=['Good'], max_workers=1)
    with (
        patch('backtesting.engine.get_registry', return_value={'Good': GoodStrategy}),
        patch('backtesting.engine.check_ticker_exists', return_value=True),
        patch('backtesting.engine.ThreadPoolExecutor', return_value=FakeExecutor([FakeFuture(GoodStrategy('1y', '1d', 'AAPL', 10_000).calculate())])),
    ):
        report = run_backtest(config)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(report)
    output = buf.getvalue()
    assert 'BACKTEST REPORT' in output
    assert 'Best:' in output and 'Worst:' in output
    assert report.results[0].ticker == 'AAPL'
