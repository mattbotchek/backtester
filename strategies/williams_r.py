"""Williams %R strategy.

Buy when Williams %R drops below -80 (oversold).
Sell when Williams %R rises above -20 (overbought).
"""

import math
from .strategy import Strategy, TradeResult


class WilliamsR(Strategy):
    name = "WilliamsR"
    description = "Buy oversold (%R<-80), sell overbought (%R>-20)."

    def __init__(self, timeline, interval, ticker, cash, period=14):
        super().__init__(timeline, interval, ticker, cash)
        self.period = period

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        high = df["High"].rolling(window=self.period).max()
        low = df["Low"].rolling(window=self.period).min()
        df["%R"] = -100 * (high - df["Close"]) / (high - low)

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["%R"]):
                continue
            price = float(row["Close"])

            if row["%R"] < -80 and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} %R={row['%R']:.1f}")
            elif row["%R"] > -20 and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} %R={row['%R']:.1f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
