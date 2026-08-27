"""Exponential Moving Average Crossover strategy.

Uses EMA instead of SMA for faster response to price changes.
"""

import math
from .strategy import Strategy, TradeResult


class EMACrossover(Strategy):
    name = "EMACrossover"
    description = "Crossover using short & long exponential moving averages."

    def __init__(self, timeline, interval, ticker, cash, short_window=12, long_window=26):
        super().__init__(timeline, interval, ticker, cash)
        self.short_window = short_window
        self.long_window = long_window

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["EMA_short"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_long"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["EMA_short"]) or math.isnan(row["EMA_long"]):
                continue
            price = float(row["Close"])

            if row["EMA_short"] > row["EMA_long"] and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} on {date}")
            elif row["EMA_short"] < row["EMA_long"] and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} on {date}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
