"""[NOVEL] Volatility Regime Adaptive strategy.

EXPERIMENTAL — This strategy is NOT based on any known published strategy.

Concept: Detects whether the market is in a low, medium, or high volatility
regime using rolling standard deviation percentiles. Uses different trading
rules per regime:
  - Low volatility: aggressive trend following (small SMA crossover)
  - Medium volatility: standard mean reversion
  - High volatility: stay in cash (risk-off)
"""

import math
from .strategy import Strategy, TradeResult


class VolatilityRegimeAdaptive(Strategy):
    name = "VolatilityRegimeAdaptive"
    description = "[NOVEL] Adapts strategy based on detected volatility regime."
    is_novel = True

    def __init__(self, timeline, interval, ticker, cash,
                 vol_window=20, vol_lookback=100,
                 low_pct=33, high_pct=66):
        super().__init__(timeline, interval, ticker, cash)
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.low_pct = low_pct
        self.high_pct = high_pct

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["Returns"] = df["Close"].pct_change()
        df["Vol"] = df["Returns"].rolling(window=self.vol_window).std()
        df["Vol_Rank"] = df["Vol"].rolling(window=self.vol_lookback).rank(pct=True) * 100

        # Trend indicators for low-vol regime
        df["SMA_fast"] = df["Close"].rolling(window=5).mean()
        df["SMA_slow"] = df["Close"].rolling(window=20).mean()

        # Mean reversion for mid-vol regime
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["ZScore"] = (df["Close"] - df["SMA_20"]) / df["Vol"] / (252 ** 0.5)

        in_position = False
        for date, row in df.iterrows():
            if math.isnan(row.get("Vol_Rank", float("nan"))):
                continue
            price = float(row["Close"])
            rank = row["Vol_Rank"]

            if rank < self.low_pct:
                # Low vol: trend follow
                if not math.isnan(row["SMA_fast"]) and not math.isnan(row["SMA_slow"]):
                    if row["SMA_fast"] > row["SMA_slow"] and not in_position:
                        self._buy(price, date=date)
                        in_position = True
                        if verbose:
                            print(f"  BUY  (low-vol trend) @ {price:.2f}")
                    elif row["SMA_fast"] < row["SMA_slow"] and in_position:
                        self._sell(price, date=date)
                        in_position = False
                        if verbose:
                            print(f"  SELL (low-vol trend) @ {price:.2f}")

            elif rank < self.high_pct:
                # Mid vol: mean reversion
                if not math.isnan(row.get("ZScore", float("nan"))):
                    if row["ZScore"] < -1.5 and not in_position:
                        self._buy(price, date=date)
                        in_position = True
                        if verbose:
                            print(f"  BUY  (mid-vol revert) @ {price:.2f}")
                    elif row["ZScore"] > 0.5 and in_position:
                        self._sell(price, date=date)
                        in_position = False
                        if verbose:
                            print(f"  SELL (mid-vol revert) @ {price:.2f}")
            else:
                # High vol: risk off
                if in_position:
                    self._sell(price, date=date)
                    in_position = False
                    if verbose:
                        print(f"  SELL (high-vol risk-off) @ {price:.2f}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
