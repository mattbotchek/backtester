"""Buy at Close, Sell at Open strategy."""

from .strategy import Strategy, TradeResult


class CloseOpen(Strategy):
    name = "BuyCloseSellOpen"
    description = "Buy at each day's close, sell at next day's open."

    def calculate(self, verbose: bool = False) -> TradeResult:
        if verbose:
            print(f"Running {self.name} backtest...\n")

        for i, (date, row) in enumerate(self.history.iterrows()):
            cur_open = float(row["Open"])
            cur_close = float(row["Close"])

            if verbose:
                print(f"  Day {i}: Open={cur_open:.2f} Close={cur_close:.2f}")

            # Sell shares at open
            if self.shares > 0:
                self._sell(cur_open, date=date)

            # Buy shares at close
            self._buy(cur_close, date=date)

        # Liquidate remaining position at final close
        if self.shares > 0:
            final_close = float(self.history.iloc[-1]["Close"])
            self._sell(final_close, date=self.history.index[-1])

        if verbose:
            print(f"\n  Final cash: {self.cash:.2f}\n")

        return self._calculate_metrics()
