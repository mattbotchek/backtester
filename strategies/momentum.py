"""Momentum strategy.

Buy when the rate of change over a lookback period exceeds a threshold.
Sell when momentum turns negative.
"""

import math
from .strategy import Strategy, TradeResult


class Momentum(Strategy):
    name = "Momentum"
    description = "Buy when N-day return exceeds threshold; sell when negative."

    def __init__(self, timeline, interval, ticker, cash, lookback=20, threshold=0.05):
        super().__init__(timeline, interval, ticker, cash)
        self.lookback = lookback
        self.threshold = threshold

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["ROC"] = df["Close"].pct_change(periods=self.lookback)

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["ROC"]):
                continue
            price = float(row["Close"])

            if row["ROC"] > self.threshold and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} ROC={row['ROC']:.3f}")
            elif row["ROC"] < 0 and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} ROC={row['ROC']:.3f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
