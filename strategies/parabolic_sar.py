"""Parabolic SAR strategy.

Buy when price crosses above the SAR.
Sell when price crosses below the SAR.
"""

from .strategy import Strategy, TradeResult


class ParabolicSAR(Strategy):
    name = "ParabolicSAR"
    description = "Trend-following using Parabolic Stop and Reverse."

    def __init__(self, timeline, interval, ticker, cash, af_start=0.02, af_step=0.02, af_max=0.2):
        super().__init__(timeline, interval, ticker, cash)
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        if len(df) < 2:
            return self._calculate_metrics()

        # Compute SAR manually
        sar = [0.0] * len(df)
        trend = 1  # 1=up, -1=down
        af = self.af_start
        ep = float(df.iloc[0]["High"])
        sar[0] = float(df.iloc[0]["Low"])

        for i in range(1, len(df)):
            high = float(df.iloc[i]["High"])
            low = float(df.iloc[i]["Low"])

            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            if trend == 1:
                if low < sar[i]:
                    trend = -1
                    sar[i] = ep
                    ep = low
                    af = self.af_start
                else:
                    if high > ep:
                        ep = high
                        af = min(af + self.af_step, self.af_max)
            else:
                if high > sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = high
                    af = self.af_start
                else:
                    if low < ep:
                        ep = low
                        af = min(af + self.af_step, self.af_max)

        df["SAR"] = sar

        in_position = False
        for date, row in df.iterrows():
            price = float(row["Close"])
            if price > row["SAR"] and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f}")
            elif price < row["SAR"] and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
