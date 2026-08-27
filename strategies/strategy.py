import yfinance
import math

class Strategy:
    def __init__(self, timeline, interval, ticker, cash):
        self.TIMELINE = timeline
        self.INTERVAL = interval
        self.TICKER = ticker
        self.INITIAL_CASH = float(cash) # too lazy to set this static for now
        self.CASH = float(cash)
        self.SHARES = 0
        
        self.TICKER_DATA = yfinance.Ticker(ticker)
        self.HISTORY = self.TICKER_DATA.history(period=timeline, interval=interval)

    # Privates

    def _getClose(self, date):
        localHistory = self.HISTORY
        openPrice = localHistory.loc[date]['Open']
        return openPrice

    def _getOpen(self, date):
        localHistory = self.HISTORY
        closePrice = localHistory.loc[date]['Close']
        return closePrice

    # Publics

    def getHistory(self, date, interval):
        history = self.TICKER_DATA.history(period=date, interval=interval)
        print(history)

        print("\n\n")