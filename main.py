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
    "Check All Tickers for Strategy",
    "Compare Ticker for Multiple Strategies"
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

def userInputChoices():
    userChoice = userChooseStrategy()

    userTimeline = input("Enter timeline (Xd/Xmo/Xy/ytd): ")
    userInterval = input("Enter interval: ")
    userCash = input("Enter starting cash: ")
    showOutput = input("Show output? (y/n): ")
    print("\n\n\n\n\n\n\n")
    
    verbose = False
    if(showOutput == 'y'):
        verbose = True

    return {"userChoice": userChoice, "userTimeline": userTimeline, "userInterval": userInterval, "userCash": userCash, "verbose": verbose}

# Probably should refactor this so that it can take config file/data sheets
def checkRunStrategy(userChoice, userTimeline, userInterval, userTicker, userCash, showOutput):
    match userChoice:
        case 0:
            strat = buyhold.BuyAndHold(userTimeline, userInterval, userTicker, userCash)
            return strat.calculate(showOutput)
        case 1:
            strat = closeopen.CloseOpen(userTimeline, userInterval, userTicker, userCash)
            return strat.calculate(showOutput)

# Check if the ticker exists,
# Boolean return
# Tests via the data length return of a ticker history request
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


# Multi ticker/strategy run work area

# Needs a big refactor, network seems to fail part way through, or maybe hitting rate limits
# Need to investigate. 
def checkAllTickers():
    choices = userInputChoices()

    # Can't tell if I suck, or if Python just gets messy
    # Probably needs a refactor
    tickerFile = open("allsymbols.txt", 'r')
    outputFile = open(f"output-{choices["userChoice"]}-{date.today()}-{choices["userTimeline"]}-{choices["userInterval"]}.csv", 'w')

    fieldNames = ["Ticker", "Final Cash Value", "ROI"]

    writer = csv.DictWriter(outputFile, fieldnames=fieldNames)
    writer.writeheader()

    for line in tickerFile:
        localTicker = line.strip()
        if(checkTickerExists(localTicker)):
            output = checkRunStrategy(choices["userChoice"], choices["userTimeline"], 
                                        choices["userInterval"], choices["localTicker"], 
                                        choices["userCash"], choices["verbose"])
            writer.writerow(output)

    tickerFile.close()
    outputFile.close()
    
    print("\n\n\n\n\n\n\n")
    main()

# Functionally this and checkAllTickers are the same, this just allows for a narrower scope
def checkMultiTicker():
    choices = userInputChoices()

    for ticker in userTicker:
        if(checkTickerExists(ticker)):
            # What is the rule? No more than 5 parameters per function? 
            output = checkRunStrategy(choices["userChoice"], choices["userTimeline"], 
                                        choices["userInterval"], choices["localTicker"], 
                                        choices["userCash"], choices["verbose"])
            print("------------------------")
            print(f"Strategy: {g_Strategies[choices["userChoice"]]}")
            print(f"Ticker: {ticker} \nFinal Cash Value: {output['Final Cash Value']} \nROI%: {output['ROI']}\n")

    print("\n\n\n\n\n\n\n")
    main()

def compareTickerForMultipleStrategies():
    strategiesToRun = []
    collectInput = True

    while collectInput:
        strategiesToRun.append(userChooseStrategy())
        continueInput = input("Select more strategies? (y/n):")

        if(continueInput == 'n'):
            collectInput = False

    localTicker = input("Enter ticker: ")
    userTimeline = input("Enter timeline (Xd/Xmo/Xy/ytd): ")
    userInterval = input("Enter interval: ")
    userCash = input("Enter starting cash: ")
    showOutput = input("Show output? (y/n): ")

    verbose = False
    if(showOutput == 'y'):
        verbose = True
    print("\n\n\n\n\n\n\n")

    for index in strategiesToRun:
        output = checkRunStrategy(index, userTimeline, 
                                        userInterval, localTicker, 
                                        userCash, verbose)
        print("------------------------")
        print(f"Strategy: {g_Strategies[index]}")
        print(f"Ticker: {localTicker} \nFinal Cash Value: {output['Final Cash Value']} \nROI%: {output['ROI']}\n")



# Compare strategy work area


def compareStrategies():
    userChoice = userChooseStrategy()


# Individual strategy work area


def individualStrategy():
    userChoices = userInputChoices()

    checkRunStrategy(choices["userChoice"], choices["userTimeline"], 
                                        choices["userInterval"], choices["localTicker"], 
                                        choices["userCash"], choices["verbose"])

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
        case 4:
            compareTickerForMultipleStrategies()

if __name__ == "__main__":
    main()