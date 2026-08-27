# ML Stock Predictor — Documentation

## Overview

The ML module uses an **ensemble model** (Random Forest + Gradient Boosting + Neural Network) to predict whether a stock will go UP, DOWN, or stay FLAT over the next N trading days, along with predicted magnitude and confidence.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Train the model (uses ~65 tickers across all sectors)
python main.py --mode ml-train

# 2. Get a prediction
python main.py --mode ml-predict -t TSLA AAPL

# 3. Run walk-forward backtest (no future data leakage)
python main.py --mode ml-backtest

# 4. Run 24/7 headless analysis
python main.py --mode ml-headless --interval 3600 --output predictions.csv
```

## Architecture

```
ml/
├── features.py    # Feature engineering (40+ technical indicators)
├── model.py       # StockPredictor ensemble model
├── backtest.py    # Walk-forward backtesting (no future leakage)
├── headless.py    # 24/7 continuous analysis daemon
└── models/        # Saved model files (.pkl)
```

### Model Components

| Component | Role |
|-----------|------|
| **Random Forest** (200 trees) | Robust baseline, handles non-linear features well |
| **Gradient Boosting** (150 trees) | Captures sequential patterns, strong on tabular data |
| **MLP Neural Network** (64→32) | Captures complex interactions between features |
| **Soft Voting** | Averages class probabilities across all 3 models |
| **GBM Regressor** | Predicts exact return magnitude |

### Features (40+)

- **Returns**: 1d, 5d, 10d, 20d percentage changes
- **Moving Averages**: SMA & EMA (5, 10, 20, 50, 200), distance from each
- **Volatility**: 10d and 20d rolling standard deviation of returns
- **RSI**: 14-period relative strength index
- **MACD**: Line, signal, histogram
- **Bollinger Bands**: Upper, lower, position within bands
- **Volume**: 20d SMA, volume ratio
- **ATR**: 14-period average true range, ATR/price ratio
- **Stochastic**: %K and %D oscillators
- **OBV**: On-balance volume + 10d SMA
- **Range**: Daily range %, gap percentage
- **Calendar**: Day of week

## How to Tune the Model

### Key Hyperparameters

Edit `ml/model.py` → `StockPredictor.__init__()`:

```python
# Classification
RandomForestClassifier(
    n_estimators=200,    # More trees = more stable but slower
    max_depth=10,        # Deeper = more complex (overfitting risk)
    min_samples_leaf=20, # Higher = more regularization
)

GradientBoostingClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,  # Lower = slower learning, better generalization
    subsample=0.8,       # <1.0 adds randomness (reduces overfitting)
)
```

### Prediction Parameters

```python
StockPredictor(
    horizon=5,       # Days ahead to predict (try 1, 5, 10, 20)
    threshold=0.02,  # 2% move = UP/DOWN (try 0.01 to 0.05)
)
```

### Feature Engineering

Add custom features in `ml/features.py` → `compute_features()`. Rules:
1. **No future data** — only use `.shift()`, `.rolling()`, `.ewm()` on past data
2. **Drop NaN rows** at the end (handled by `.dropna()`)
3. Add the column name to the returned DataFrame

## Preventing Overfitting

### Built-in Safeguards

1. **Time-Series Cross-Validation**: Uses `TimeSeriesSplit` — never trains on future data
2. **Walk-Forward Backtesting**: Expanding window that simulates real deployment
3. **Regularization**: `min_samples_leaf=20`, `max_depth` limits, `subsample<1`
4. **Early Stopping**: MLP uses `early_stopping=True`
5. **Diverse Ensemble**: 3 different model types reduce single-model overfitting
6. **Broad Training Data**: 65+ tickers across all sectors prevents ticker-specific fitting

### Signs of Overfitting

| Symptom | Solution |
|---------|----------|
| Training accuracy >> CV accuracy | Reduce `max_depth`, increase `min_samples_leaf` |
| Great on training tickers, poor on new ones | Add more tickers, reduce features |
| Model memorizes recent patterns | Use longer `timeline` (5y+), more CV splits |
| Single feature dominates | Check `feature_importance()`, remove or combine |

### Manual Checks

```python
from ml.model import StockPredictor

predictor = StockPredictor()
predictor.load()

# Check feature importances — top features shouldn't dominate
print(predictor.feature_importance().head(20))

# Compare training stats
print(predictor._training_stats)
# cv_accuracy_std should be < 0.05 (stable across folds)
# If accuracy > 0.6, be suspicious (markets are ~50/50)
```

### Cross-Validation Interpretation

- **Good**: CV accuracy 0.40-0.55, low std (< 0.03) — model learns some real signal
- **Suspicious**: CV accuracy > 0.60 — likely overfitting to some pattern
- **Bad**: CV accuracy < 0.33 — model worse than random (3-class)

## Headless / 24/7 Mode

```bash
# Default: analyze 65 stocks every hour, print to console
python main.py --mode ml-headless

# Custom tickers, 5-minute intervals, save to CSV
python main.py --mode ml-headless \
    --tickers AAPL TSLA NVDA \
    --interval 300 \
    --output predictions.csv

# Or run the module directly
python -m ml.headless --model stock_predictor --interval 600
```

The headless mode:
- Loads a pre-trained model
- Cycles through all tickers, generating predictions
- Prints summary with high-confidence calls
- Optionally appends to CSV for later analysis
- Handles `Ctrl+C` gracefully
- Runs until manually stopped

## Output Format

Each prediction includes:

```
AAPL ↑ +1.23% (confidence: 72%) Outliers: RSI extremely overbought (82)
TSLA ↓ -2.45% (confidence: 65%)
MSFT → +0.12% (confidence: 45%)
```

| Field | Description |
|-------|-------------|
| **Direction** | UP (↑), DOWN (↓), or FLAT (→) |
| **Predicted Return %** | Expected magnitude over the horizon period |
| **Confidence** | Highest class probability (0-100%) |
| **Outliers** | Extreme indicator values that may affect reliability |

## Walk-Forward Backtest

```bash
python main.py --mode ml-backtest
```

This runs an expanding-window backtest:
1. Train on data up to time T
2. Predict T to T+horizon
3. Expand training window, repeat

**No future data is ever seen during training.** Results show:
- Classification accuracy (UP/DOWN/FLAT)
- Directional accuracy (ignoring FLAT predictions)
- Mean absolute error on return magnitude
- Full classification report with precision/recall/F1

## Important Disclaimers

⚠️ **This is for educational/research purposes only.**

- Stock markets are inherently unpredictable
- Past performance does not guarantee future results
- The model may find spurious correlations
- Always validate with walk-forward backtesting before any real use
- Never invest money you can't afford to lose based on model output
