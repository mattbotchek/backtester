"""Bollinger Bands strategy.

Buy when price touches the lower band (oversold).
Sell when price touches the upper band (overbought).
"""

import math
from .strategy import Strategy, TradeResult


class BollingerBands(Strategy):
    name = "BollingerBands"
    description = "Mean reversion using Bollinger Bands."

    def __init__(self, timeline, interval, ticker, cash, window=20, num_std=2):
        super().__init__(timeline, interval, ticker, cash)
        self.window = window
        self.num_std = num_std

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["SMA"] = df["Close"].rolling(window=self.window).mean()
        df["STD"] = df["Close"].rolling(window=self.window).std()
        df["Upper"] = df["SMA"] + (df["STD"] * self.num_std)
        df["Lower"] = df["SMA"] - (df["STD"] * self.num_std)

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["SMA"]):
                continue
            price = float(row["Close"])

            if price <= row["Lower"] and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} (below lower band {row['Lower']:.2f})")
            elif price >= row["Upper"] and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} (above upper band {row['Upper']:.2f})")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
