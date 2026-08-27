#
# Backtesting random strategies:
# This is the main entry point file, you enter here, choose your strategy
# Then enter relevant information for that strategy (length of time, ticker, etc)
# Then allow the tester to run and output relevant information
#

import yfinance
from strategies import closeopen, buyhold
from datetime import date
import csv
import logging

# Globals

# Don't know how globals are treated in Python, or rather looked at by reviewers
# I'm probably doing this wrong, so refactor to use... dict? Can I do a dict that points to a function? 
# Functions to add:
# Check single ticker against all strategies
g_Modes = [
    "Individual Strategy",
    "Compare Strategy",
    "Check Multiple Tickers for Strategy",
    "Check All Tickers for Strategy"
]

# Strategies I think that would be cool to implement:
# Non quantitative strategies (ie, not data run, but assigning non quantitative values to a stock)
# Buy Open Sell Close <- Self explanatory
# 200 Day Moving Average <- Buy when price is below 200 day moving average, sell when above
g_Strategies = [
    "BuyAndHold",
    "BuyCloseSellOpen"
]


# Helper functions


def userChooseStrategy():
    print("Enter choice for back testing:\n")

    i = 0
    for strategy in g_Strategies:
        print(f"{i}) {strategy}\n")
        i=i+1

    userChoice = int(input("Choice: "))

    if(userChoice > i-1 or userChoice < 0):
        print("Not an option\n")
        main()

    return userChoice

# Probably should refactor this so that it can take config file/data sheets
def checkRunStrategy(userChoice, userTimeline, userInterval, userTicker, userCash, showOutput):
    match userChoice:
        case 0:
            strat = buyhold.BuyAndHold(userTimeline, userInterval, userTicker, userCash)
            return strat.calculate(showOutput)
        case 1:
            strat = closeopen.CloseOpen(userTimeline, userInterval, userTicker, userCash)
            return strat.calculate(showOutput)

def checkTickerExists(ticker):
    yfinanceLogger = logging.getLogger("yfinance")
    wasDisabled = yfinanceLogger.disabled
    yfinanceLogger.disabled = True

    try:
        history = yfinance.Ticker(ticker).history(period="1d")
        return not history.empty
    except Exception:
        return False
    finally:
        yfinanceLogger.disabled = wasDisabled


# Multi ticker run work area

# Needs a big refactor, network seems to fail part way through, or maybe hitting rate limits
# Need to investigate. 
def checkAllTickers():
    # Below "questions" chunk needs to be refactored into a function, but this is rough first draft
    userChoice = userChooseStrategy()

    userTimeline = input("Enter timeline (Xd/Xmo/Xy/ytd): ")
    userInterval = input("Enter interval: ")
    userCash = input("Enter starting cash: ")
    showOutput = input("Show output? (y/n): ")
    print("\n\n\n\n\n\n\n")
    
    verbose = False
    if(showOutput == 'y'):
        verbose = True

    # Can't tell if I suck, or if Python just gets messy
    # Probably needs a refactor
    tickerFile = open("allsymbols.txt", 'r')
    outputFile = open(f"output-{userChoice}-{date.today()}-{userTimeline}-{userInterval}.csv", 'w')
    fieldNames = ["Ticker", "Final Cash Value", "ROI"]
    writer = csv.DictWriter(outputFile, fieldnames=fieldNames)
    writer.writeheader()

    for line in tickerFile:
        localTicker = line.strip()
        if(checkTickerExists(localTicker)):
            output = checkRunStrategy(userChoice, userTimeline, userInterval, localTicker, userCash, verbose)
            writer.writerow(output)

    tickerFile.close()
    outputFile.close()
    
    print("\n\n\n\n\n\n\n")
    main()

# Functionally this and checkAllTickers are the same, this just allows for a narrower scope
def checkMultiTicker():
    # Needs refactor into function
    userChoice = userChooseStrategy()

    userTicker = input("Enter stock tickers (comma-separated): ")
    userTicker = userTicker.split(",")
    userTimeline = input("Enter timeline (Xd/Xmo/Xy/ytd): ")
    userInterval = input("Enter interval: ")
    userCash = input("Enter starting cash: ")
    showOutput = input("Show output? (y/n): ")
    print("\n\n\n\n\n\n\n")
    verbose = False

    if(showOutput == 'y'):
        verbose = True

    for ticker in userTicker:
        if(checkTickerExists(ticker)):
            # What is the rule? No more than 5 parameters per function? 
            output = checkRunStrategy(userChoice, userTimeline, userInterval, ticker, userCash, verbose)
            print(f"Ticker: {ticker} \nFinal Cash Value: {output['Final Cash Value']} \nROI%: {output['ROI']}\n")

    print("\n\n\n\n\n\n\n")
    main()

    

# Compare strategy work area


def compareStrategies():
    userChoice = userChooseStrategy()


# Individual strategy work area


def individualStrategy():
    # Needs refactor here too
    userChoice = userChooseStrategy()

    userTimeline = input("Enter timeline (Xd/Xmo/Xy/ytd): ")
    userInterval = input("Enter interval: ")
    userTicker = input("Enter stock tickers (comma-separated): ")
    userTicker = userTicker.split(",")
    print(f"Tickers: {userTicker}\n")
    userCash = input("Enter starting cash: ")
    print("\n\n\n\n\n\n\n")

    checkRunStrategy(userChoice, userTimeline, userInterval, userTicker, userCash)

    print("\n\n\n\n\n\n\n")
    main()


# Main function


def main():
    print("Choose Mode: ")

    i = 0
    for mode in g_Modes:
        print(f"{i}) {mode}\n")
        i=i+1

    userChoice = int(input("Choice: "))

    if(userChoice > i-1 or userChoice < 0):
        print("Not an option\n")
        main()

    match userChoice:
        case 0:
            individualStrategy()
        case 1:
            compareStrategies()
        case 2:
            checkMultiTicker()
        case 3:
            checkAllTickers()

if __name__ == "__main__":
    main()