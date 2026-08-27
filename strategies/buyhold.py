import yfinance
import math
from .strategy import Strategy

class BuyAndHold(Strategy):
    def calculate(self, showOutput):
        if(showOutput):
            print("Running backtest calculation...\n\n")

        localData = self.HISTORY.iloc[[0, -1]]

        initialBuy = float(localData.iloc[0]["Open"])
    
        self.SHARES = math.floor(self.CASH/initialBuy)
        self.CASH = self.CASH - (self.SHARES * initialBuy)

        if(showOutput):
            print(f"Initial Shares: {self.SHARES}")
            print(f"Initial Cash: {self.CASH}\n\n")

        sell = float(localData.iloc[1]["Close"])
        self.CASH = self.CASH + (self.SHARES * sell)
        self.SHARES = 0

        if(showOutput):
            print(f"Final Shares: {self.SHARES}")
            print(f"Final Cash: {self.CASH}\n\n")

        roi = float(((self.CASH / self.INITIAL_CASH) * 100) - 100)

        return {"Ticker": self.TICKER, "Final Cash Value": self.CASH, "ROI": roi}
