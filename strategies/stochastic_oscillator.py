"""Stochastic Oscillator strategy.

Buy when %K crosses above %D in oversold territory (<20).
Sell when %K crosses below %D in overbought territory (>80).
"""

import math
from .strategy import Strategy, TradeResult


class StochasticOscillator(Strategy):
    name = "StochasticOscillator"
    description = "Trade on %K/%D crossovers in extreme zones."

    def __init__(self, timeline, interval, ticker, cash,
                 k_period=14, d_period=3, oversold=20, overbought=80):
        super().__init__(timeline, interval, ticker, cash)
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        low_min = df["Low"].rolling(window=self.k_period).min()
        high_max = df["High"].rolling(window=self.k_period).max()
        df["%K"] = ((df["Close"] - low_min) / (high_max - low_min)) * 100
        df["%D"] = df["%K"].rolling(window=self.d_period).mean()

        in_position = False
        prev_k, prev_d = None, None
        for date, row in df.iterrows():
            if math.isnan(row["%K"]) or math.isnan(row["%D"]):
                continue
            price = float(row["Close"])
            k, d = row["%K"], row["%D"]

            if prev_k is not None:
                # Bullish crossover in oversold
                if prev_k <= prev_d and k > d and k < self.oversold and not in_position:
                    self._buy(price, date=date)
                    in_position = True
                    if verbose:
                        print(f"  BUY  @ {price:.2f} %K={k:.1f}")
                # Bearish crossover in overbought
                elif prev_k >= prev_d and k < d and k > self.overbought and in_position:
                    self._sell(price, date=date)
                    in_position = False
                    if verbose:
                        print(f"  SELL @ {price:.2f} %K={k:.1f}")

            prev_k, prev_d = k, d

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
