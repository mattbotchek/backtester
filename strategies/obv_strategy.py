"""On-Balance Volume (OBV) strategy.

Buy when OBV SMA crosses above its longer SMA (volume confirms uptrend).
Sell when OBV SMA crosses below.
"""

import math
import numpy as np
from .strategy import Strategy, TradeResult


class OBVStrategy(Strategy):
    name = "OBV"
    description = "Trade on On-Balance Volume moving average crossovers."

    def __init__(self, timeline, interval, ticker, cash, short_window=10, long_window=30):
        super().__init__(timeline, interval, ticker, cash)
        self.short_window = short_window
        self.long_window = long_window

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()

        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])

        df["OBV"] = obv
        df["OBV_short"] = df["OBV"].rolling(window=self.short_window).mean()
        df["OBV_long"] = df["OBV"].rolling(window=self.long_window).mean()

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["OBV_short"]) or math.isnan(row["OBV_long"]):
                continue
            price = float(row["Close"])

            if row["OBV_short"] > row["OBV_long"] and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f}")
            elif row["OBV_short"] < row["OBV_long"] and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
