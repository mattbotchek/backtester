"""Average True Range (ATR) Breakout strategy.

Buy when price breaks above the previous close + ATR multiplier.
Sell when price breaks below the previous close - ATR multiplier.
Uses volatility to determine breakout significance.
"""

import math
import pandas as pd
from .strategy import Strategy, TradeResult


class ATRBreakout(Strategy):
    name = "ATRBreakout"
    description = "Breakout entries based on ATR volatility bands."

    def __init__(self, timeline, interval, ticker, cash, atr_period=14, multiplier=1.5):
        super().__init__(timeline, interval, ticker, cash)
        self.atr_period = atr_period
        self.multiplier = multiplier

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=self.atr_period).mean()

        in_position = False
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            date = df.index[i]

            if math.isnan(row["ATR"]):
                continue
            price = float(row["Close"])
            prev_close = float(prev["Close"])
            upper = prev_close + self.multiplier * row["ATR"]
            lower = prev_close - self.multiplier * row["ATR"]

            if price > upper and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} (breakout above {upper:.2f})")
            elif price < lower and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} (breakdown below {lower:.2f})")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
