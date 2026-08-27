"""[NOVEL] Intraday Range Compression strategy.

EXPERIMENTAL — This strategy is NOT based on any known published strategy.

Concept: When the daily High-Low range compresses to historically low levels
(relative to recent history), a breakout is imminent. Buys when range contracts
below the 10th percentile of the last 50 days AND price closes in the upper
half of the range. Sells when range expands above the 80th percentile (the
breakout has played out) or when a stop-loss is triggered.
"""

import math
from .strategy import Strategy, TradeResult


class RangeCompression(Strategy):
    name = "RangeCompression"
    description = "[NOVEL] Buys range compression, sells expansion breakout."
    is_novel = True

    def __init__(self, timeline, interval, ticker, cash,
                 lookback=50, compress_pct=10, expand_pct=80):
        super().__init__(timeline, interval, ticker, cash)
        self.lookback = lookback
        self.compress_pct = compress_pct
        self.expand_pct = expand_pct

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["Range"] = df["High"] - df["Low"]
        df["Range_Rank"] = df["Range"].rolling(window=self.lookback).rank(pct=True) * 100
        df["Close_Pos"] = (df["Close"] - df["Low"]) / df["Range"]

        in_position = False
        entry_price = 0.0
        for date, row in df.iterrows():
            if math.isnan(row.get("Range_Rank", float("nan"))) or row["Range"] == 0:
                continue
            price = float(row["Close"])

            if row["Range_Rank"] < self.compress_pct and row["Close_Pos"] > 0.5 and not in_position:
                self._buy(price, date=date)
                in_position = True
                entry_price = price
                if verbose:
                    print(f"  BUY  @ {price:.2f} (range compressed, rank={row['Range_Rank']:.0f})")
            elif in_position:
                # Exit on expansion or 5% stop loss
                if row["Range_Rank"] > self.expand_pct or price < entry_price * 0.95:
                    self._sell(price, date=date)
                    in_position = False
                    if verbose:
                        reason = "expansion" if row["Range_Rank"] > self.expand_pct else "stop-loss"
                        print(f"  SELL @ {price:.2f} ({reason})")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
