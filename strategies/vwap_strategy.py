"""Volume Weighted Average Price (VWAP) strategy.

Buy when price is below VWAP (undervalued).
Sell when price is above VWAP (overvalued).
"""

import math
from .strategy import Strategy, TradeResult


class VWAPStrategy(Strategy):
    name = "VWAP"
    description = "Buy below VWAP, sell above VWAP."

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["Cum_Vol"] = df["Volume"].cumsum()
        df["Cum_VP"] = (df["Close"] * df["Volume"]).cumsum()
        df["VWAP"] = df["Cum_VP"] / df["Cum_Vol"]

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row["VWAP"]) or row["Cum_Vol"] == 0:
                continue
            price = float(row["Close"])

            if price < row["VWAP"] and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} VWAP={row['VWAP']:.2f}")
            elif price > row["VWAP"] and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} VWAP={row['VWAP']:.2f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
