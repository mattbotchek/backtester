# Machine Learning Guide

The application has three prediction families that share one durable lifecycle:

- `StockPredictor`: the default daily direction and target-price model.
- `HorizonModel`: real-time 5-minute, intraday, 1-day, 5-day, 20-day,
  60-day, and 252-day models.
- `CombinedPredictor`: quantitative, news-sentiment, and alternative-data model.

Every live prediction records when it was made, the market data it used, when its
forecast is due, its target price, the eventual observed price, and whether the
directional forecast was correct. All timestamps are stored as UTC ISO-8601.

## Install

```bash
pip install -r requirements.txt
```

Optional NVIDIA CUDA acceleration uses XGBoost:

```bash
pip install -r requirements-gpu.txt
```

GPU estimators are selected only when XGBoost and an NVIDIA GPU are both
available. Set `ML_ACCELERATOR=cpu` to force CPU estimators. Set
`ML_IO_WORKERS` to override the bounded worker count used for independent data
downloads and predictions.

## Run the complete system

Start the FastAPI web application and the coordinated collection, prediction,
settlement, and retraining service:

```bash
python run_server.py
```

Open `http://localhost:8000`.

Useful options:

```bash
python run_server.py --port 8080
python run_server.py --predict-interval 30 --retrain-hours 6
python run_server.py --target-accuracy 0.70 --minimum-accuracy 0.65
python run_server.py --minimum-outcomes 50
python run_server.py --no-scraper
python run_server.py --no-predictions
python run_server.py --no-retrain
```

The continuous service keeps running after reaching the configured threshold so
that new public information is collected, due forecasts are settled, drift is
measured, and all model families continue to be refreshed.

Continuous training runs in a separate process from Uvicorn and reserves two
CPU cores by default, so model work cannot monopolize page-serving capacity.
Set `ML_RESERVED_CPUS` or `ML_CPU_WORKERS` to tune that budget. Web requests
serve predictions and sentiment from their durable ledgers. Live quotes use a
persistent stale-while-refresh cache with a 3.5-second cold-load budget, and
charts load Plotly after the initial page instead of blocking render.

Run the continuous pipeline without the web server:

```bash
python main.py --mode ml-continuous
python main.py --mode ml-continuous --interval 1800 --retrain-hours 6
python main.py --mode ml-continuous --target-accuracy 0.70 --minimum-accuracy 0.65 --minimum-outcomes 30
```

## Prediction lifecycle

`ml/prediction_store.py` is the source of truth for model registrations,
training times, predictions, settlements, and accuracy.

Each persisted prediction includes:

| Field | Meaning |
|---|---|
| Model and training version | Identifies the model family and exact training run |
| Ticker | Predicted security |
| `created_at` | When the prediction was generated |
| `input_time` | Timestamp of the newest market bar available to the model |
| `target_time` | When the advertised forecast is expected to be true |
| Interval and horizon | Forecast cadence and number of bars |
| Base price | Price known when the forecast was made |
| Predicted return and target price | Forecast magnitude |
| Direction and confidence | `UP`, `DOWN`, or `FLAT` and model probability |
| Actual price and observation time | First eligible market observation at or after the target |
| Status and hit | Pending/settled state and directional result |

Daily and intraday targets use the XNYS exchange calendar, including holidays
and early closes. The advertised target is the time when the complete target
bar becomes observable, not the bar's opening timestamp. A forecast is not
scored until that completion time passes and an eligible completed market bar
exists. Intraday models do not issue forecasts when their horizon would cross
the current session boundary, matching their session-grouped training labels.

The directional rule is:

- `UP` is a hit when the actual price is at or above the predicted target price.
- `DOWN` is a hit when the actual price is at or below the predicted target price.
- `FLAT` is a hit when the actual price is within 0.25% of the target price.

Accuracy is the mean hit rate over settled forecasts only. Pending predictions
never enter the denominator. Retraining creates a new model version, so a new
forecast from the same market bar is retained rather than overwriting an older
model's forecast.

By default, lifecycle data is stored in
`ml/models/prediction_history.db`. Set `PREDICTION_DB_PATH` to override it.

## Web views and APIs

The stock detail page shows every available model family. Prediction cards
include creation time, input time, target time, target price, last training
time, settled sample count, and live directional accuracy.

Daily and intraday charts overlay:

- the price known when the model predicted;
- the model's target price at the advertised target time; and
- the eventual actual price after the outcome becomes observable.

Prediction history remains available after settlement and identifies pending,
hit, and miss outcomes.

Relevant endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/stock/{ticker}/predictions` | Current predictions from all model families |
| `GET /api/predictions/history` | Filterable durable prediction history |
| `POST /api/predictions/settle` | Settle all currently due outcomes |
| `GET /api/models/status` | Training timestamps and live forward statistics |
| `GET /api/stock/{ticker}/intraday` | Intraday bars and prediction overlays |

The dashboard, market-wide screen, watchlist, stock-detail page, and ML control
screen use the same lifecycle data rather than independent accuracy claims.

## Accuracy: production gate versus research diagnostics

The production target is measured only from genuine prospective forecasts:

1. The model persists a prediction before its result exists.
2. The advertised target time passes.
3. The settlement process obtains the first eligible market price.
4. The ledger applies the directional hit rule.
5. `ContinuousTrainingService` evaluates the settled outcomes.

The default target is 70%. The configured fallback floor is 65%, and the score
is not accepted until at least `--minimum-outcomes` forecasts have settled.
Service state is persisted so restarts do not reset progress.

Training CV and historical backtests remain useful diagnostics, but neither can
satisfy the live production gate. In particular:

- `ml-train` reports temporal cross-validation.
- `ml-backtest` and `ml-horizons-backtest` report historical simulations.
- `ml/optimize_loop.py` compares temporal CV and unseen-ticker holdouts for
  offline model selection; its holdout score is not live future accuracy.

The `realtime_5m` model uses an untouched final two-trading-day holdout and the
same target-price settlement rule as production. It predicts `UP` or `DOWN`
only for moves of at least 0.25%; smaller moves are explicitly `FLAT`, with the
same 0.25% tolerance used by the ledger. Intraday training targets are grouped
by market session so they never cross an overnight or weekend boundary.

The web application labels these metrics separately and uses only settled
prediction-ledger outcomes for the live accuracy statistic.

## Data and point-in-time safety

`ml/alternative_data.py` collects public observations into
`ml/models/alternative_data.db`. Supported sources include:

- CNBC, MarketWatch, Reuters, and configurable RSS news;
- SEC submissions, filing metadata, and Form 4 insider transactions;
- configurable congressional transaction feeds;
- configurable executive or presidential public RSS feeds;
- USGS earthquake reports;
- NASA EONET natural-disaster reports;
- GDELT geopolitical events;
- yfinance news, fundamentals, earnings, and options; and
- arbitrary standardized JSON feeds.

These features are joined centrally in `compute_features()` and
`compute_intraday_features()`, so the default, horizon, combined, optimization,
and backtest paths receive the same fixed alternative-data schema.

Every observation has:

- `event_time`: when the event happened;
- `available_at`: when the public source made it available; and
- `collected_at`: when this process retrieved it.

A historical feature query includes an observation only when both
`event_time <= as_of` and `available_at <= as_of`. This prevents future
information from leaking into an earlier training row. Ticker queries include
that ticker and global events, but never observations belonging only to another
ticker.

Fresh installations have sparse historical alternative-data coverage. The
fixed schema supplies zero-valued features before eligible observations exist,
then fills naturally as the permanent collector accumulates point-in-time data.

### Alternative-data configuration

| Variable | Purpose |
|---|---|
| `ALT_DATA_DB_PATH` | Override the SQLite database path |
| `ALT_DATA_HTTP_TIMEOUT_SECONDS` | HTTP timeout for collectors |
| `ALT_DATA_SEC_USER_AGENT` | Truthful SEC user agent with contact information |
| `ALT_DATA_SEC_RATE_LIMIT_SECONDS` | Delay between SEC requests |
| `ALT_DATA_SEC_MAX_FILINGS_PER_TICKER` | Filing records collected per ticker |
| `ALT_DATA_SEC_MAX_FORM4_DOCUMENTS` | Form 4 documents parsed per ticker |
| `ALT_DATA_NEWS_RSS_FEEDS` | Additional news RSS URLs |
| `ALT_DATA_EXECUTIVE_RSS_FEEDS` | Public executive/presidential RSS URLs |
| `ALT_DATA_CONGRESSIONAL_ENDPOINTS` | Congressional transaction endpoints |
| `ALT_DATA_CUSTOM_JSON_FEEDS` | Standardized custom JSON endpoints |
| `ALT_DATA_ENABLE_USGS` | Enable or disable USGS collection |
| `ALT_DATA_ENABLE_EONET` | Enable or disable NASA EONET collection |
| `ALT_DATA_ENABLE_GDELT` | Enable or disable GDELT collection |

Feed variables accept a JSON object mapping source names to URLs, a JSON list of
URLs, or comma-separated URLs. Sources without a stable official
unauthenticated API remain opt-in rather than being scraped through brittle or
unauthorized paths.

## Features

The common feature pipeline includes:

- Multi-window returns, volatility, volume, and momentum.
- SMA/EMA distances, RSI, MACD, Bollinger bands, ATR, stochastic oscillators,
  OBV, VWAP, and calendar signals.
- ADX, CCI, MFI, Chaikin money flow, Williams %R, Donchian and Keltner
  channels, Ichimoku values, drawdown, downside volatility, autocorrelation,
  and price-volume correlation.
- Market breadth, yield-curve context, commodities, cryptocurrencies,
  beta/correlation, VIX, sector ETFs, bonds, and the US dollar.
- Point-in-time news, filings, insider transactions, public-official trading,
  disasters, geopolitical events, fundamentals, earnings, and options.
- Sentiment from article headlines, Fear & Greed data, and ticker-specific
  news for the combined model.

Feature selection, scaling, and estimators are fitted only from the appropriate
training partition. Training samples are globally time ordered before temporal
splits.

## Model commands

Train and save the default model:

```bash
python main.py --mode ml-train
python main.py --mode ml-train --tickers AAPL MSFT NVDA --timeline 5y
```

Load it and create persisted live predictions:

```bash
python main.py --mode ml-predict --tickers AAPL MSFT
```

Run historical evaluation:

```bash
python main.py --mode ml-backtest
python main.py --mode ml-horizons-backtest
```

Train all horizon models:

```bash
python main.py --mode ml-horizons-train
python main.py --mode ml-horizons-train --horizon realtime_5m intraday short_1d short_5d medium_20d long_60d long_252d
```

Run the default model as a standalone prediction daemon:

```bash
python main.py --mode ml-headless --tickers AAPL MSFT --interval 3600
```

## Storage

Generated runtime artifacts live under `ml/models/`:

| Path | Contents |
|---|---|
| `ml/models/prediction_history.db` | Model registrations, immutable predictions, outcomes, and accuracy |
| `ml/models/alternative_data.db` | Point-in-time public observations and source health |
| `ml/models/continuous_training_state.json` | Continuous-service progress and gate state |
| `ml/models/stock_predictor.pkl` | Default trained model |
| `ml/models/model_*.pkl` | Multi-horizon trained models |
| `ml/models/combined_model.pkl` | Combined model |

Do not commit generated databases or model artifacts.

## Important use note

Predictions are probabilistic research outputs, not financial advice. A 70%
live directional target is a measured operating goal, not a guarantee for any
individual trade. The durable ledger makes the result auditable and prevents
pending, backfilled, CV, or historical outcomes from being reported as future
production performance.
