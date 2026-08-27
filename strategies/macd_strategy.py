"""MACD (Moving Average Convergence Divergence) strategy.

Buy when MACD line crosses above the signal line.
Sell when MACD line crosses below the signal line.
"""

import math
from .strategy import Strategy, TradeResult


class MACDStrategy(Strategy):
    name = "MACD"
    description = "Trade on MACD / signal line crossovers."

    def __init__(self, timeline, interval, ticker, cash,
                 fast=12, slow=26, signal=9):
        super().__init__(timeline, interval, ticker, cash)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["EMA_fast"] = df["Close"].ewm(span=self.fast, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.slow, adjust=False).mean()
        df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
        df["Signal"] = df["MACD"].ewm(span=self.signal_period, adjust=False).mean()

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["MACD"]) or math.isnan(row["Signal"]):
                continue
            price = float(row["Close"])

            if row["MACD"] > row["Signal"] and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f}")
            elif row["MACD"] < row["Signal"] and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
