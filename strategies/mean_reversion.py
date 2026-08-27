"""Mean Reversion strategy.

Assumes price reverts to its moving average.
Buy when price deviates significantly below the mean.
Sell when it returns to or exceeds the mean.
"""

import math
from .strategy import Strategy, TradeResult


class MeanReversion(Strategy):
    name = "MeanReversion"
    description = "Buy when price is >2 std devs below SMA, sell at mean."

    def __init__(self, timeline, interval, ticker, cash, window=20, threshold=2.0):
        super().__init__(timeline, interval, ticker, cash)
        self.window = window
        self.threshold = threshold

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["SMA"] = df["Close"].rolling(window=self.window).mean()
        df["STD"] = df["Close"].rolling(window=self.window).std()
        df["ZScore"] = (df["Close"] - df["SMA"]) / df["STD"]

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["ZScore"]):
                continue
            price = float(row["Close"])

            if row["ZScore"] < -self.threshold and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} z={row['ZScore']:.2f}")
            elif row["ZScore"] >= 0 and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} z={row['ZScore']:.2f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
