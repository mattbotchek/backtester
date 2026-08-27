"""[NOVEL] Volume Shock Reversal strategy.

EXPERIMENTAL — This strategy is NOT based on any known published strategy.

Concept: Detects "volume shocks" — days where volume is >2x the rolling average.
On a volume shock with a price drop, assumes panic selling and buys the reversal.
On a volume shock with a price surge, assumes euphoria exhaustion and sells.
The idea is that extreme volume events tend to mark local extremes.
"""

import math
from .strategy import Strategy, TradeResult


class VolumeShockReversal(Strategy):
    name = "VolumeShockReversal"
    description = "[NOVEL] Contrarian trades on extreme volume spikes."
    is_novel = True

    def __init__(self, timeline, interval, ticker, cash,
                 vol_window=20, shock_multiplier=2.0, price_drop_pct=-0.02):
        super().__init__(timeline, interval, ticker, cash)
        self.vol_window = vol_window
        self.shock_multiplier = shock_multiplier
        self.price_drop_pct = price_drop_pct

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["Vol_SMA"] = df["Volume"].rolling(window=self.vol_window).mean()
        df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA"]
        df["Return"] = df["Close"].pct_change()

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row.get("Vol_Ratio", float("nan"))):
                continue
            price = float(row["Close"])
            is_shock = row["Vol_Ratio"] > self.shock_multiplier

            if is_shock and row["Return"] < self.price_drop_pct and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} (vol shock + drop)")
            elif is_shock and row["Return"] > abs(self.price_drop_pct) and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} (vol shock + surge)")
            elif not is_shock and in_position and row["Return"] > 0.03:
                # Take profits on moderate gains without shock
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} (profit take)")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
