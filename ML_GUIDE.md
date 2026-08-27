# ML Stock Predictor — Documentation

## Overview

The ML module uses an **ensemble model** (Random Forest + HistGradientBoosting + Neural Network) with **120+ features** including cross-asset context (SPY, VIX, sector ETFs, treasuries, credit spreads) to predict whether a stock will go UP or DOWN over configurable horizons.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Train the default model (uses ~70 tickers across all sectors)
python main.py --mode ml-train

# 2. Get a prediction
python main.py --mode ml-predict -t TSLA AAPL

# 3. Run walk-forward backtest (no future data leakage)
python main.py --mode ml-backtest

# 4. Run 24/7 headless analysis
python main.py --mode ml-headless --interval 3600 --output predictions.csv

# 5. Train multi-horizon models (intraday through 1-year)
python main.py --mode ml-horizons-train --horizon short_1d short_5d medium_20d long_60d

# 6. Backtest multi-horizon models
python main.py --mode ml-horizons-backtest --horizon short_5d medium_20d

# 7. Run accuracy optimization loop
python -m ml.optimize_loop --horizon 5 --target 0.70 --rounds 3
```

## Architecture

```
ml/
├── features.py        # Feature engineering (120+ indicators + cross-asset)
├── model.py           # StockPredictor ensemble model
├── multi_horizon.py   # Multi-horizon model system (1d through 252d)
├── optimize_loop.py   # Automated accuracy optimization
├── backtest.py        # Walk-forward backtesting (no future leakage)
├── headless.py        # 24/7 continuous analysis daemon
└── models/            # Saved model files (.pkl)
```

### Multi-Horizon Models

| Horizon | Name | Use Case | Interval |
|---------|------|----------|----------|
| Intraday | `intraday` | Day trading | 5-minute bars |
| 1 day | `short_1d` | Overnight swing trades | Daily |
| 5 days | `short_5d` | Weekly swing trades | Daily |
| 20 days | `medium_20d` | Monthly position trades | Daily |
| 60 days | `long_60d` | Quarterly positions | Daily |
| 252 days | `long_252d` | Annual investment decisions | Weekly |

### Features (120+)

- **Returns**: Multi-period (1d through 20d) percentage changes
- **Moving Averages**: SMA & EMA (5, 10, 20, 50, 200), distance ratios, crossover signals
- **Volatility**: Multi-period rolling std, Parkinson volatility, volatility ratios
- **RSI**: Multi-period (7, 14, 21) with divergence detection
- **MACD**: Line, signal, histogram, crossover detection
- **Bollinger Bands**: Multi-period (20, 50), position within bands
- **Volume**: SMA, ratio, OBV, OBV slope, volume-price trend
- **ATR**: 14-period average true range, ATR/price ratio
- **Stochastic**: %K and %D oscillators
- **Calendar**: Day of week, month, quarter, month start/end
- **Streak**: Consecutive up/down day counts
- **Lag**: 1-5 day lagged returns and volume ratios
- **Statistical**: Skewness, kurtosis over rolling windows
- **Feature Interactions**: RSI × volume, momentum × volatility, SMA spread × RSI

#### Cross-Asset Context Features

| Data Source | Features Added |
|-------------|---------------|
| SPY (S&P 500) | 1d/5d/20d returns, SMA ratio |
| ^VIX (Volatility) | Level, 10d SMA, change, 1d/5d/20d returns |
| QQQ (Nasdaq 100) | 1d/5d/20d returns, SMA ratio |
| IWM (Russell 2000) | 1d/5d/20d returns, SMA ratio |
| TLT (20yr Treasury) | 1d/5d/20d returns, SMA ratio |
| GLD (Gold) | 1d/5d/20d returns, SMA ratio |
| UUP (US Dollar) | 1d/5d/20d returns, SMA ratio |
| HYG (High Yield) | 1d/5d/20d returns, SMA ratio |
| Sector ETFs (XLK, XLF, etc.) | Relative strength, sector returns |
| Market breadth | Risk-on/off (tech vs utilities) |
| Credit spread | HYG vs TLT momentum |

## Tested Accuracy Results

### Walk-Forward Backtest (Temporal — Honest Results)

The gold-standard test: train on PAST data only, predict FUTURE unseen data.
Uses rolling retraining (2-year window, quarterly steps) across 70 tickers.

| Horizon | Overall Accuracy | High-Confidence (>75%) |
|---------|-----------------|------------------------|
| 1 day | 50.3% | 56.6% (2% of preds) |
| 5 days | 50.8% | 57.8% (8% of preds) |
| 20 days | 52.0% | 52.3% (17% of preds) |
| 60 days | 52.8% | 53.9% (31% of preds) |

### Cross-Sectional Holdout (Train on some tickers, test on others)

Train on 54 tickers, test on 15 held-out tickers (same time period).
Tests generalization to unseen stocks within the same market regime.

| Horizon | Accuracy |
|---------|----------|
| 1 day | 67.3% |
| 5 days | 66.6% |
| 20 days | 65.6% |
| 60 days | 66.4% |

### Cross-Validation (Within Training Data)

| Horizon | CV Accuracy |
|---------|-------------|
| 1 day | 62.4% ± 2.9% |
| 5 days | 60.9% ± 3.2% |
| 20 days | 61.2% ± 4.4% |
| 60 days | 60.4% ± 7.3% |

### Why Temporal Walk-Forward Is Lower

The temporal test is the most realistic but hardest:
- **Market regimes change**: Patterns from 2019-2022 may not predict 2023-2025
- **EMH applies**: Markets are approximately efficient — consistent prediction is hard
- **No alternative data**: Free technical indicators are already priced in by most participants
- **Cross-sectional scores are inflated**: Training and test data share the same macro environment

## How to Improve Accuracy

### What Won't Help (Diminishing Returns)
- More trees / deeper models — accuracy plateaus around 60% CV
- More technical indicators — already exhaustive (120+)
- Feature selection — top features are cross-asset, which are essential

### What WOULD Help (Requires Investment)

| Data Source | Expected Improvement | Cost |
|-------------|---------------------|------|
| **News Sentiment** (Refinitiv, Bloomberg) | +3-5% | $$$$ |
| **Social Sentiment** (StockTwits, Reddit NLP) | +2-4% | $-$$ |
| **Fundamental Data** (Earnings, analyst revisions) | +3-5% | $-$$ |
| **Options Flow** (Put/call ratios, IV surfaces) | +2-3% | $$$ |
| **Alternative Data** (Satellite, web traffic) | +1-3% | $$$$ |
| **Insider Trading** (SEC EDGAR filings) | +1-2% | Free (parsing effort) |

### Practical Strategies

1. **Confidence Filtering**: Only act on high-confidence predictions (>65%)
2. **Sector-Specific Models**: Train per-sector for tighter patterns
3. **Regime Detection**: Use VIX levels to switch between bull/bear models
4. **Ensemble of Horizons**: Combine signals across timeframes
5. **Rolling Retraining**: Update model weekly with most recent data

## Preventing Overfitting

### Built-in Safeguards

1. **Time-Series Cross-Validation**: Uses `TimeSeriesSplit` — never trains on future data
2. **Walk-Forward Backtesting**: Expanding window simulating real deployment
3. **Regularization**: `min_samples_leaf=30`, `max_depth` limits, `l2_regularization`
4. **Early Stopping**: HistGBM and MLP both use early stopping
5. **Rolling Retraining**: 2-year window prevents memorizing ancient patterns
6. **Broad Training Data**: 70+ tickers across all sectors

### Signs of Overfitting

| Symptom | Solution |
|---------|----------|
| CV accuracy >> walk-forward accuracy | Reduce `max_depth`, increase regularization |
| CV accuracy > 65% | Very suspicious — check for data leakage |
| One feature dominates importance | Remove or cap that feature |
| Model performs perfectly in one regime | Test across multiple market conditions |

## Headless / 24/7 Mode

```bash
# Default: analyze 65 stocks every hour
python main.py --mode ml-headless

# Custom tickers, 5-minute intervals, save to CSV
python main.py --mode ml-headless --tickers AAPL TSLA NVDA --interval 300 --output predictions.csv
```

## Important Disclaimers

⚠️ **This is for educational/research purposes only.**

- Stock markets are approximately efficient — consistent 70%+ accuracy requires alternative data
- The temporal walk-forward results (50-53%) reflect realistic production expectations
- Higher cross-sectional scores (66-67%) reflect the model's ability to distinguish stocks within a market regime
- Never invest money you can't afford to lose based on model output
- Past performance does not guarantee future results
