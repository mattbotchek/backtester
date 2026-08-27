"""Ichimoku Cloud strategy.

Buy when price crosses above the cloud (Senkou Span A > B).
Sell when price crosses below the cloud.
"""

import math
from .strategy import Strategy, TradeResult


class Ichimoku(Strategy):
    name = "Ichimoku"
    description = "Trade based on Ichimoku Cloud signals."

    def __init__(self, timeline, interval, ticker, cash,
                 tenkan=9, kijun=26, senkou_b=52):
        super().__init__(timeline, interval, ticker, cash)
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b_period = senkou_b

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()

        high_t = df["High"].rolling(window=self.tenkan).max()
        low_t = df["Low"].rolling(window=self.tenkan).min()
        df["Tenkan"] = (high_t + low_t) / 2

        high_k = df["High"].rolling(window=self.kijun).max()
        low_k = df["Low"].rolling(window=self.kijun).min()
        df["Kijun"] = (high_k + low_k) / 2

        df["SpanA"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(self.kijun)

        high_b = df["High"].rolling(window=self.senkou_b_period).max()
        low_b = df["Low"].rolling(window=self.senkou_b_period).min()
        df["SpanB"] = ((high_b + low_b) / 2).shift(self.kijun)

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row.get("SpanA", float("nan"))) or math.isnan(row.get("SpanB", float("nan"))):
                continue
            price = float(row["Close"])
            cloud_top = max(row["SpanA"], row["SpanB"])
            cloud_bottom = min(row["SpanA"], row["SpanB"])

            if price > cloud_top and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} (above cloud)")
            elif price < cloud_bottom and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} (below cloud)")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
