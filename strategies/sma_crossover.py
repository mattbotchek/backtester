"""Simple Moving Average Crossover strategy.

Buy when the short-term SMA crosses above the long-term SMA (golden cross).
Sell when the short-term SMA crosses below the long-term SMA (death cross).
"""

from .strategy import Strategy, TradeResult


class SMACrossover(Strategy):
    name = "SMACrossover"
    description = "Golden/death cross using short & long simple moving averages."

    def __init__(self, timeline, interval, ticker, cash, short_window=50, long_window=200):
        super().__init__(timeline, interval, ticker, cash)
        self.short_window = short_window
        self.long_window = long_window

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()
        df["SMA_short"] = df["Close"].rolling(window=self.short_window).mean()
        df["SMA_long"] = df["Close"].rolling(window=self.long_window).mean()

        in_position = False
        for date, row in df.iterrows():
            if row["SMA_short"] is None or row["SMA_long"] is None:
                continue
            import math
            if math.isnan(row["SMA_short"]) or math.isnan(row["SMA_long"]):
                continue

            price = float(row["Close"])

            if row["SMA_short"] > row["SMA_long"] and not in_position:
                self._buy(price, date=date)
                in_position = True
                if verbose:
                    print(f"  BUY  @ {price:.2f} on {date}")
            elif row["SMA_short"] < row["SMA_long"] and in_position:
                self._sell(price, date=date)
                in_position = False
                if verbose:
                    print(f"  SELL @ {price:.2f} on {date}")

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
