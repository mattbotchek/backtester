"""Buy and Hold strategy — buy at first open, sell at last close."""

import math

from .strategy import Strategy, TradeResult


class BuyAndHold(Strategy):
    name = "BuyAndHold"
    description = "Buy at the first open price and hold until the last close price."

    def calculate(self, verbose: bool = False) -> TradeResult:
        if verbose:
            print(f"Running {self.name} backtest...\n")

        first_open = float(self.history.iloc[0]["Open"])
        last_close = float(self.history.iloc[-1]["Close"])

        self._buy(first_open, date=self.history.index[0])

        if verbose:
            print(f"  Bought {self.shares} shares @ {first_open:.2f}")

        self._sell(last_close, date=self.history.index[-1])

        if verbose:
            print(f"  Sold all shares @ {last_close:.2f}")
            print(f"  Final cash: {self.cash:.2f}\n")

        return self._calculate_metrics()
