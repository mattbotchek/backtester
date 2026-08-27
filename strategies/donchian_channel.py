"""Donchian Channel Breakout strategy (Turtle Trading).

Buy when price breaks above the N-day high.
Sell when price breaks below the N-day low.
"""

import math
from .strategy import Strategy, TradeResult


class DonchianChannel(Strategy):
    name = "DonchianChannel"
    description = "Turtle trading — buy N-day high breakout, sell N-day low."

    def __init__(self, timeline, interval, ticker, cash, entry_period=20, exit_period=10):
        super().__init__(timeline, interval, ticker, cash)
        self.entry_period = entry_period
        self.exit_period = exit_period

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["Upper"] = df["High"].rolling(window=self.entry_period).max()
        df["Lower"] = df["Low"].rolling(window=self.exit_period).min()

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["Upper"]) or math.isnan(row["Lower"]):
                continue
            price = float(row["Close"])

            if price > row["Upper"] and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f}")
            elif price < row["Lower"] and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
