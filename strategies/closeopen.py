import yfinance
import math
from .strategy import Strategy

class CloseOpen(Strategy):
    def calculate(self, showOutput):
        if(showOutput):
            print("Running backtest calculation...\n\n")

        iDay = 0
        curClose = 0
        curOpen = 0
        for day in self.HISTORY.itertuples(index=False):
            if(showOutput):
                print(f"Day: {iDay}")
                print(f"Cash: {self.CASH}")
                print(f"Shares: {self.SHARES}")
            curOpen = float(day.Open)
            curClose = float(day.Close)
            if(showOutput):
                print(f"Open: {curOpen}")
                print(f"Close: {curClose}")
                print("--------------------------")
                print("\n\n")

            if(self.SHARES != 0):
                self.CASH = self.CASH + self.SHARES * curOpen
                self.SHARES = 0

            self.SHARES = math.floor(self.CASH/curClose)
            self.CASH = self.CASH - self.SHARES * curClose
            iDay = iDay + 1

        self.CASH = self.CASH + self.SHARES * curClose
        if(showOutput):
            print(self.CASH)

        roi = float((self.CASH / self.INITIAL_CASH) * 100)
        return {"Ticker": self.TICKER, "Final Cash Value": self.CASH, "ROI": roi}
        
