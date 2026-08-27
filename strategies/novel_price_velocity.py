"""[NOVEL] Price Velocity Divergence strategy.

EXPERIMENTAL — This strategy is NOT based on any known published strategy.

Concept: Computes "price velocity" (rate of change of rate of change — i.e.,
the second derivative of price). When velocity is accelerating but price is
near a support level, it signals a strong move is imminent. Buys on positive
acceleration from low velocity; sells when acceleration turns sharply negative
from high velocity.
"""

import math
from .strategy import Strategy, TradeResult


class PriceVelocityDivergence(Strategy):
    name = "PriceVelocityDivergence"
    description = "[NOVEL] Trades on price acceleration (2nd derivative) signals."
    is_novel = True

    def __init__(self, timeline, interval, ticker, cash, roc_period=5, accel_threshold=0.002):
        super().__init__(timeline, interval, ticker, cash)
        self.roc_period = roc_period
        self.accel_threshold = accel_threshold

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["ROC"] = df["Close"].pct_change(periods=self.roc_period)
        df["Acceleration"] = df["ROC"].diff()

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row.get("Acceleration", float("nan"))):
                continue
            price = float(row["Close"])
            accel = row["Acceleration"]
            roc = row["ROC"]

            # Buy: acceleration turning positive from negative ROC
            if accel > self.accel_threshold and roc < 0 and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} accel={accel:.4f} roc={roc:.4f}")
            # Sell: acceleration turning negative from positive ROC
            elif accel < -self.accel_threshold and roc > 0 and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} accel={accel:.4f} roc={roc:.4f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
