"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import math
import logging

import yfinance
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Standardized result from a strategy backtest run."""

    ticker: str
    strategy_name: str
    initial_cash: float
    final_cash: float
    roi_percent: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    trade_log: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def to_dict(self) -> dict:
        return {
            "Ticker": self.ticker,
            "Strategy": self.strategy_name,
            "Initial Cash": self.initial_cash,
            "Final Cash Value": self.final_cash,
            "ROI": self.roi_percent,
            "Total Trades": self.total_trades,
            "Win Rate": f"{self.win_rate:.2%}",
            "Max Drawdown": f"{self.max_drawdown_percent:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio:.4f}",
        }


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement `calculate()` and set `name`.
    Strategies are auto-discovered by the registry if placed in
    the strategies/ package.
    """

    name: str = "BaseStrategy"
    description: str = ""
    # Mark experimental/novel strategies
    is_novel: bool = False

    def __init__(self, timeline: str, interval: str, ticker: str, cash: float):
        self.timeline = timeline
        self.interval = interval
        self.ticker = ticker
        self.initial_cash = float(cash)
        self.cash = float(cash)
        self.shares = 0
        self.trade_log: list[dict] = []

        self._ticker_data = yfinance.Ticker(ticker)
        self.history: pd.DataFrame = self._ticker_data.history(
            period=timeline, interval=interval
        )

    def _buy(self, price: float, date=None, fraction: float = 1.0) -> int:
        """Buy as many shares as affordable at `price`. Returns shares bought."""
        affordable = math.floor((self.cash * fraction) / price)
        if affordable <= 0:
            return 0
        cost = affordable * price
        self.shares += affordable
        self.cash -= cost
        self.trade_log.append(
            {"action": "BUY", "price": price, "shares": affordable, "date": str(date)}
        )
        return affordable

    def _sell(self, price: float, date=None, shares: Optional[int] = None) -> float:
        """Sell shares at `price`. Returns proceeds."""
        to_sell = shares if shares is not None else self.shares
        to_sell = min(to_sell, self.shares)
        if to_sell <= 0:
            return 0.0
        proceeds = to_sell * price
        self.shares -= to_sell
        self.cash += proceeds
        self.trade_log.append(
            {"action": "SELL", "price": price, "shares": to_sell, "date": str(date)}
        )
        return proceeds

    def _calculate_metrics(self) -> TradeResult:
        """Compute standard performance metrics from trade log."""
        roi = ((self.cash / self.initial_cash) * 100) - 100

        buys = [t for t in self.trade_log if t["action"] == "BUY"]
        sells = [t for t in self.trade_log if t["action"] == "SELL"]
        total_trades = min(len(buys), len(sells))
        winning = 0
        losing = 0
        for b, s in zip(buys, sells):
            if s["price"] > b["price"]:
                winning += 1
            else:
                losing += 1

        # Max drawdown from equity curve
        max_drawdown = 0.0
        if len(self.history) > 0:
            prices = self.history["Close"].values
            peak = prices[0]
            for p in prices:
                if p > peak:
                    peak = p
                dd = ((peak - p) / peak) * 100
                if dd > max_drawdown:
                    max_drawdown = dd

        # Simplified Sharpe ratio (annualized daily returns)
        sharpe = 0.0
        if len(self.history) > 1:
            returns = self.history["Close"].pct_change().dropna()
            if returns.std() != 0:
                sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)

        return TradeResult(
            ticker=self.ticker,
            strategy_name=self.name,
            initial_cash=self.initial_cash,
            final_cash=self.cash,
            roi_percent=roi,
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            max_drawdown_percent=max_drawdown,
            sharpe_ratio=sharpe,
            trade_log=self.trade_log,
        )

    @abstractmethod
    def calculate(self, verbose: bool = False) -> TradeResult:
        """Run the strategy and return a TradeResult."""
        ...

    def __repr__(self) -> str:
        tag = " [NOVEL]" if self.is_novel else ""
        return f"{self.name}{tag}: {self.description}"