import io
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd

from strategies import TradeResult, get_registry, list_strategies
from strategies.atr_breakout import ATRBreakout
from strategies.bollinger_bands import BollingerBands
from strategies.donchian_channel import DonchianChannel
from strategies.novel_gap_fade import GapFade
from strategies.novel_volume_shock import VolumeShockReversal
from strategies.sma_crossover import SMACrossover
from strategies.strategy import Strategy
from tests.conftest import FakeYFTicker, make_ohlcv


class DummyBaseStrategy(Strategy):
    name = 'DummyBase'
    description = 'exercise base class helpers'

    def calculate(self, verbose: bool = False) -> TradeResult:
        self._buy(10.0, date='2026-01-01')
        self._sell(12.0, date='2026-01-02', shares=1)
        self._sell(11.0, date='2026-01-03')
        return self._calculate_metrics()


def _fake_ticker(data: pd.DataFrame):
    return FakeYFTicker('AAPL', data=data)


def _date_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range('2026-01-01', periods=n, freq='B', tz='UTC')


def _atr_breakout_data() -> pd.DataFrame:
    idx = _date_index(20)
    close = [100.0] * 14 + [150.0, 120.0, 80.0, 90.0, 91.0, 92.0]
    high = [101.0] * 14 + [151.0, 121.0, 81.0, 91.0, 92.0, 93.0]
    low = [99.0] * 14 + [149.0, 119.0, 79.0, 89.0, 90.0, 91.0]
    return pd.DataFrame(
        {
            'Open': close,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': [1_000_000.0] * 20,
        },
        index=idx,
    )


def _donchian_breakout_data() -> pd.DataFrame:
    idx = _date_index(15)
    close = [100.0] * 10 + [130.0, 95.0, 96.0, 97.0, 98.0]
    high = [101.0] * 10 + [120.0, 96.0, 97.0, 98.0, 99.0]
    low = [99.0] * 10 + [100.0, 110.0, 95.0, 96.0, 97.0]
    return pd.DataFrame(
        {
            'Open': close,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': [900_000.0] * 15,
        },
        index=idx,
    )


def _bollinger_final_sell_data() -> pd.DataFrame:
    idx = _date_index(30)
    close = [100.0] * 19 + [80.0] + [81.0] * 10
    return pd.DataFrame(
        {
            'Open': close,
            'High': [c + 1 for c in close],
            'Low': [c - 1 for c in close],
            'Close': close,
            'Volume': [1_100_000.0] * 30,
        },
        index=idx,
    )


def _gap_fill_data() -> pd.DataFrame:
    idx = _date_index(4)
    return pd.DataFrame(
        {
            'Open': [100.0, 90.0, 92.0, 80.0],
            'High': [101.0, 101.0, 93.0, 81.0],
            'Low': [99.0, 89.0, 79.0, 70.0],
            'Close': [100.0, 100.0, 80.0, 70.0],
            'Volume': [1_000_000.0] * 4,
        },
        index=idx,
    )


def _sma_cross_data() -> pd.DataFrame:
    idx = _date_index(8)
    close = [10.0, 10.0, 10.0, 20.0, 20.0, 5.0, 5.0, 5.0]
    return pd.DataFrame(
        {
            'Open': close,
            'High': [c + 0.5 for c in close],
            'Low': [c - 0.5 for c in close],
            'Close': close,
            'Volume': [500_000.0] * 8,
        },
        index=idx,
    )


def _volume_shock_data() -> pd.DataFrame:
    idx = _date_index(26)
    close = [100.0] * 20 + [90.0, 94.0, 94.0, 110.0, 108.0, 109.0]
    volume = [1_000_000.0] * 20 + [3_500_000.0, 1_100_000.0, 1_100_000.0, 4_000_000.0, 1_100_000.0, 1_100_000.0]
    return pd.DataFrame(
        {
            'Open': [100.0] * 20 + [92.0, 94.0, 94.0, 108.0, 108.0, 109.0],
            'High': [101.0] * 20 + [93.0, 95.0, 95.0, 111.0, 109.0, 110.0],
            'Low': [99.0] * 20 + [89.0, 93.0, 93.0, 107.0, 107.0, 108.0],
            'Close': close,
            'Volume': volume,
        },
        index=idx,
    )


def test_registry_discovers_all_strategies(mock_yfinance):
    registry = get_registry()
    assert registry
    assert sorted(registry) == list_strategies()
    for name, strategy_cls in registry.items():
        strategy = strategy_cls('1y', '1d', 'AAPL', 10_000)
        assert name == strategy.name
        assert strategy.ticker == 'AAPL'
        assert isinstance(repr(strategy), str)


@patch('yfinance.Ticker', return_value=_fake_ticker(make_ohlcv(n=260, seed=7)))
def test_every_strategy_calculates_with_shared_fixture(_patched):
    for strategy_cls in get_registry().values():
        result = strategy_cls('1y', '1d', 'AAPL', 10_000).calculate(verbose=True)
        assert isinstance(result, TradeResult)
        assert result.ticker == 'AAPL'
        assert isinstance(result.to_dict(), dict)


@patch('yfinance.Ticker', return_value=_fake_ticker(make_ohlcv(n=30, seed=11)))
def test_base_strategy_helpers_cover_metrics_and_zero_share_paths(_patched):
    dummy = DummyBaseStrategy('1y', '1d', 'AAPL', 100.0)
    assert dummy._buy(1_000.0) == 0
    assert dummy._sell(10.0) == 0.0
    result = dummy.calculate()
    assert result.total_trades == 1
    assert result.winning_trades == 1
    assert result.losing_trades == 0
    assert result.win_rate == 1.0
    assert 'ROI' in result.to_dict()
    assert '[NOVEL]' not in repr(dummy)


@patch('yfinance.Ticker', return_value=_fake_ticker(_atr_breakout_data()))
def test_atr_breakout_verbose_buy_and_sell(_patched):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = ATRBreakout('1y', '1d', 'AAPL', 10_000, atr_period=14, multiplier=1.0).calculate(verbose=True)
    output = buf.getvalue()
    assert 'BUY' in output and 'SELL' in output
    assert result.total_trades >= 1


@patch('yfinance.Ticker', return_value=_fake_ticker(_donchian_breakout_data()))
def test_donchian_breakout_verbose_buy_and_sell(_patched):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = DonchianChannel('1y', '1d', 'AAPL', 10_000, entry_period=10, exit_period=2).calculate(verbose=True)
    output = buf.getvalue()
    assert output.count('BUY') == 1
    assert 'SELL' in output
    assert result.total_trades >= 1


@patch('yfinance.Ticker', return_value=_fake_ticker(_bollinger_final_sell_data()))
def test_bollinger_final_sell_of_remaining_position(_patched):
    result = BollingerBands('1y', '1d', 'AAPL', 10_000, window=20, num_std=1).calculate(verbose=True)
    sells = [trade for trade in result.trade_log if trade['action'] == 'SELL']
    assert sells, result.trade_log
    assert sells[-1]['date'] == str(_bollinger_final_sell_data().index[-1])


@patch('yfinance.Ticker', return_value=_fake_ticker(_gap_fill_data()))
def test_gap_fade_gap_fill_and_stop_loss_paths(_patched):
    fill_result = GapFade('1y', '1d', 'AAPL', 10_000, gap_threshold=0.05).calculate(verbose=True)
    assert any(trade['action'] == 'SELL' for trade in fill_result.trade_log)

    stop_data = _gap_fill_data().copy()
    stop_data.iloc[1, stop_data.columns.get_loc('Close')] = 85.0
    with patch('yfinance.Ticker', return_value=_fake_ticker(stop_data)):
        stop_result = GapFade('1y', '1d', 'AAPL', 10_000, gap_threshold=0.05).calculate(verbose=True)
    assert len([trade for trade in stop_result.trade_log if trade['action'] == 'SELL']) >= 1


@patch('yfinance.Ticker', return_value=_fake_ticker(_volume_shock_data()))
def test_volume_shock_branches_and_verbose_output(_patched):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = VolumeShockReversal('1y', '1d', 'AAPL', 10_000).calculate(verbose=True)
    output = buf.getvalue()
    assert 'vol shock + drop' in output
    assert ('vol shock + surge' in output) or ('profit take' in output)
    assert result.total_trades >= 1


@patch('yfinance.Ticker', return_value=_fake_ticker(_sma_cross_data()))
def test_sma_crossover_buy_and_sell_verbose(_patched):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = SMACrossover('1y', '1d', 'AAPL', 10_000, short_window=2, long_window=3).calculate(verbose=True)
    output = buf.getvalue()
    assert output.count('BUY') == 1
    assert output.count('SELL') == 1
    assert result.total_trades == 1
