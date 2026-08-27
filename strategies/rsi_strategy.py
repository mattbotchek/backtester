"""Relative Strength Index (RSI) strategy.

Buy when RSI drops below the oversold threshold (30).
Sell when RSI rises above the overbought threshold (70).
"""

import math
from .strategy import Strategy, TradeResult


class RSIStrategy(Strategy):
    name = "RSI"
    description = "Buy oversold (RSI<30), sell overbought (RSI>70)."

    def __init__(self, timeline, interval, ticker, cash, period=14,
                 oversold=30, overbought=70):
        super().__init__(timeline, interval, ticker, cash)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["RSI"]):
                continue
            price = float(row["Close"])

            if row["RSI"] < self.oversold and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} RSI={row['RSI']:.1f}")
            elif row["RSI"] > self.overbought and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} RSI={row['RSI']:.1f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
