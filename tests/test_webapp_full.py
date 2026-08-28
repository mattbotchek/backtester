import asyncio
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from webapp import app as webapp
from webapp.cache import PersistentTTLCache
from tests.conftest import FakeYFTicker, make_intraday_ohlcv, make_ohlcv


@pytest.fixture()
def client(isolated_env, mock_yfinance, tmp_path, monkeypatch):
    cache = PersistentTTLCache(tmp_path / 'web-cache.json', max_workers=2)
    monkeypatch.setattr(webapp, '_web_cache', cache)
    with TestClient(webapp.app) as client:
        yield client
    cache.close()


def test_training_tracker_and_helper_functions(monkeypatch, isolated_env, mock_yfinance, tmp_path):
    tracker = webapp.TrainingTracker()
    tracker.start('default', total_tickers=2)
    tracker.update('default', 1)
    tracker.finish('default', accuracy=0.8)
    tracker.start('bad', total_tickers=1)
    tracker.fail('bad', 'oops')
    assert tracker.get('default')['state'] == 'ready'
    assert tracker.get_all()['bad']['state'] == 'error'

    history = webapp._get_stock_history('AAPL')
    metadata = webapp._get_stock_metadata('AAPL')
    assert history['ticker'] == 'AAPL'
    assert metadata['market_cap'] > 0

    monkeypatch.setattr(webapp, 'get_registry', lambda: {'ok': lambda **kwargs: None})
    class FakeStrategy:
        name = 'Strategy'
        is_novel = True
        def __init__(self, timeline, interval, ticker, cash):
            pass
        def calculate(self):
            return SimpleNamespace(roi_percent=5.0, win_rate=0.5, total_trades=2)
    monkeypatch.setattr(webapp, 'get_registry', lambda: {'s': FakeStrategy})
    assert webapp._run_strategies('AAPL')

    class FakePredictor:
        horizon = 1
        def load(self):
            return None
        def predict(self, ticker):
            return SimpleNamespace(model_name='stock_predictor', direction='UP', predicted_return_pct=1.0, confidence=0.8, outliers=[], timestamp='2026-01-01T00:00:00Z', input_at='2026-01-01T00:00:00Z', predicted_for='2026-01-02T00:00:00Z', base_price=100.0, predicted_price=101.0, last_trained_at='2026-01-01T00:00:00Z', forward_accuracy=0.75, settled_predictions=4)
    monkeypatch.setitem(__import__('sys').modules, 'ml.model', SimpleNamespace(StockPredictor=FakePredictor))
    prediction = webapp._get_ml_prediction('AAPL')
    assert prediction['direction'] == 'UP'
    monkeypatch.setitem(__import__('sys').modules, 'ml.model', SimpleNamespace(StockPredictor=lambda: (_ for _ in ()).throw(RuntimeError('no model'))))
    assert webapp._get_ml_prediction('AAPL')['model_status'] == 'not_available'

    normalized = webapp._normalize_model_prediction({'predicted_return': 0.1234, 'confidence': 0.88, 'forward_accuracy': 0.65, 'base_price': 100.12345, 'predicted_price': 101.98765})
    assert normalized['confidence'] == 88.0
    stored = webapp._stored_prediction({'model_name': 'm', 'ticker': 'AAPL', 'direction': 'UP', 'predicted_return_pct': 1.0, 'confidence': 0.8, 'metadata': {'outliers': ['x']}, 'created_at': '2026-01-01T00:00:00Z', 'input_at': '2026-01-01T00:00:00Z', 'target_at': '2026-01-02T00:00:00Z', 'base_price': 100.0, 'predicted_price': 101.0, 'model_trained_at': '2026-01-01T00:00:00Z', 'interval': '1d', 'horizon': 1, 'status': 'pending'}, {'m': {'forward_accuracy': 0.7, 'settled_count': 3}})
    assert stored['forward_accuracy'] == 70.0


def test_webapp_routes_and_startup(monkeypatch, client, isolated_env):
    store = webapp.PredictionStore()
    store.record_prediction(model_name='stock_predictor', family='StockPredictor', ticker='AAPL', input_at='2099-01-01T00:00:00Z', target_at='2099-01-02T00:00:00Z', interval='1d', horizon=1, base_price=100, predicted_return_pct=1, direction='UP', confidence=0.8, last_trained_at='2099-01-01T00:00:00Z')
    store.record_prediction(model_name='model_intraday', family='HorizonModel', ticker='AAPL', input_at='2099-01-01T00:00:00Z', target_at='2099-01-01T00:05:00Z', interval='5m', horizon=5, base_price=100, predicted_return_pct=0.5, direction='UP', confidence=0.7, last_trained_at='2099-01-01T00:00:00Z')

    webapp.training_tracker._status.clear()
    monkeypatch.setattr(webapp, '_get_all_ml_predictions', lambda ticker: ([{'model_name': 'stock_predictor', 'horizon': 1, 'direction': 'UP', 'predicted_return': 1.0, 'confidence': 80.0}], []))
    monkeypatch.setattr(webapp, '_auto_train_models', lambda: None)
    monkeypatch.setitem(__import__('sys').modules, 'ml.sentiment.scrapers', SimpleNamespace(get_aggregated_sentiment=lambda: {'fear_greed_index': 55}))

    for path in ['/', '/stock/AAPL', '/strategies', '/ml', '/picks', '/market']:
        response = client.get(path)
        assert response.status_code == 200

    assert client.get('/api/tickers').json()['tickers']
    assert client.get('/api/tickers?q=abcd').json()['tickers'][0] == 'ABCD'
    assert 'popular' in client.get('/api/popular').json()
    assert client.get('/api/stock/AAPL').json()['ticker'] == 'AAPL'
    assert 'strategies' in client.get('/api/stock/AAPL/strategies').json()
    assert 'prediction' in client.get('/api/stock/AAPL/prediction').json()
    assert 'predictions' in client.get('/api/stock/AAPL/predictions').json()
    assert 'predictions' in client.get('/api/predictions/history').json()
    assert 'due' in client.post('/api/predictions/settle').json()
    assert 'models' in client.get('/api/models/status').json()
    assert 'indices' in client.get('/api/market/overview').json()
    assert 'market' in client.get('/api/market/predictions').json()
    assert 'predictions' in client.get('/api/predictions/daily').json()
    assert 'models' in client.get('/api/training/status').json()
    assert client.post('/api/training/start').json()['status'] == 'started'
    webapp.training_tracker.start('default', total_tickers=1)
    assert client.post('/api/training/start').json()['status'] == 'already_training'
    assert client.get('/api/sentiment').json()['sentiment']['fear_greed_index'] == 55
    intraday = client.get('/api/stock/AAPL/intraday?period=1d&interval=5m').json()
    assert intraday['ticker'] == 'AAPL'
    assert 'model_status' in client.get('/api/stock/AAPL/daytrading/predict').json() or 'direction' in client.get('/api/stock/AAPL/daytrading/predict').json()
    assert client.post('/api/daytrading/train').json()['status'] == 'started'
    webapp.training_tracker.start('intraday', total_tickers=1)
    assert client.post('/api/daytrading/train').json()['status'] == 'already_training'
    assert client.get('/api/timeframes').json()['default'] == webapp.DEFAULT_TIMEFRAME
    assert webapp.create_app() is webapp.app


def test_schedule_refresh_and_slow_request_warning(monkeypatch, isolated_env, tmp_path):
    webapp._prediction_refreshes.clear()
    monkeypatch.setattr(webapp, '_get_all_ml_predictions', lambda ticker: ([{'ticker': ticker}], []))
    assert webapp._schedule_prediction_refresh('AAPL') is True
    assert webapp._schedule_prediction_refresh('AAPL') is True
    webapp._prediction_refreshes.update({str(i) for i in range(20)})
    assert webapp._schedule_prediction_refresh('MSFT') is False
    webapp._prediction_refreshes.clear()

    values = iter([0.0, 6.0])
    monkeypatch.setattr(webapp.time, 'perf_counter', lambda: next(values))
    warnings = []
    monkeypatch.setattr(webapp.logger, 'warning', lambda *args: warnings.append(args))
    async def call_next(request):
        from fastapi.responses import JSONResponse
        return JSONResponse({'ok': True})
    request = SimpleNamespace(method='GET', url=SimpleNamespace(path='/slow'))
    response = asyncio.run(webapp.add_request_timing(request, call_next))
    assert 'Server-Timing' in response.headers
    assert warnings



def test_cache_states_model_loading_and_error_routes(monkeypatch, isolated_env, tmp_path, mock_yfinance):
    cache = PersistentTTLCache(tmp_path / 'web-cache-extra.json', max_workers=1)
    try:
        monkeypatch.setattr(webapp, '_web_cache', cache)
        assert webapp._cached_payload('k1', lambda: {'value': 1}, ttl_seconds=60, fallback={})['cache_state'] == 'loaded'
        assert webapp._cached_payload('k1', lambda: {'value': 2}, ttl_seconds=60, fallback={})['cache_state'] == 'fresh'
        cache._entries['old'] = {'updated_at': 0, 'value': {'cached': True}}
        stale = webapp._cached_payload('old', lambda: {'cached': False}, ttl_seconds=1, fallback={})
        assert stale['cache_state'] == 'stale'

        model_dir = tmp_path / 'models'
        model_dir.mkdir(exist_ok=True)
        for filename in ['stock_predictor.pkl', 'combined_model.pkl', 'model_short_1d.pkl']:
            (model_dir / filename).write_text('x', encoding='utf-8')
        monkeypatch.setitem(__import__('sys').modules, 'ml.multi_horizon', SimpleNamespace(HORIZONS={'short_1d': SimpleNamespace(model_file='model_short_1d')}, HorizonModel=lambda config: SimpleNamespace(load=lambda: None, predict_ticker=lambda ticker: {'model_name': 'model_short_1d', 'horizon': 1, 'direction': 'UP', 'predicted_return': 1.0, 'confidence': 0.8}), MODEL_DIR=model_dir))
        monkeypatch.setitem(__import__('sys').modules, 'ml.sentiment.combined_model', SimpleNamespace(CombinedPredictor=lambda: SimpleNamespace(load=lambda: None, predict=lambda ticker: {'model_name': 'combined_model', 'horizon': 5, 'direction': 'DOWN', 'predicted_return': -1.0, 'confidence': 0.6})))
        monkeypatch.setattr(webapp, '_get_ml_prediction', lambda ticker: {'model_name': 'stock_predictor', 'horizon': 0, 'direction': 'UP', 'predicted_return': 0.5, 'confidence': 0.9})
        monkeypatch.setattr(webapp, '_settle_due_predictions', lambda force=False: {'status': 'started'})
        preds, errors = webapp._get_all_ml_predictions('AAPL')
        assert len(preds) == 3 and errors == []

        class TrainOK:
            def train(self, tickers, timeline='5y'):
                return {'cv_accuracy_mean': 0.8}
            def save(self):
                return None
        monkeypatch.setitem(__import__('sys').modules, 'ml.model', SimpleNamespace(StockPredictor=lambda: TrainOK()))
        webapp.training_tracker._status.clear()
        webapp._auto_train_models()
        assert webapp.training_tracker.get('default')['state'] == 'ready'
        monkeypatch.setitem(__import__('sys').modules, 'ml.model', SimpleNamespace(StockPredictor=lambda: (_ for _ in ()).throw(RuntimeError('broken'))))
        webapp._auto_train_models()
        assert webapp.training_tracker.get('default')['state'] == 'error'

        monkeypatch.setattr(webapp.yfinance, 'Ticker', lambda ticker: SimpleNamespace(history=lambda **kwargs: pd.DataFrame()))
        intraday = asyncio.run(webapp.api_intraday('AAPL', period='1d', interval='5m'))
        assert 'error' in intraday
        monkeypatch.setattr(webapp, '_schedule_prediction_refresh', lambda ticker: False)
        day = asyncio.run(webapp.api_daytrading_predict('AAPL'))
        assert day['model_status'] == 'not_trained'
        monkeypatch.setitem(__import__('sys').modules, 'ml.sentiment.scrapers', SimpleNamespace(get_aggregated_sentiment=lambda: (_ for _ in ()).throw(RuntimeError('sentiment fail'))))
        assert 'error' in asyncio.run(webapp.api_sentiment())
    finally:
        cache.close()


def test_persistent_ttl_cache_states_and_failures(tmp_path, monkeypatch):
    bad_path = tmp_path / 'invalid.json'
    bad_path.write_text('{broken', encoding='utf-8')
    cache = PersistentTTLCache(bad_path, max_workers=1, max_entries=1)
    try:
        value, state = cache.get('fresh', lambda: {'x': 1}, ttl_seconds=60, fallback={})
        assert state == 'loaded' and value == {'x': 1}
        value, state = cache.get('fresh', lambda: {'x': 2}, ttl_seconds=60, fallback={})
        assert state == 'fresh'

        class SlowFuture:
            def result(self, timeout=None):
                from concurrent.futures import TimeoutError
                raise TimeoutError()
        monkeypatch.setattr(cache, '_refresh', lambda key, loader: SlowFuture())
        value, state = cache.get('new', lambda: {'x': 3}, ttl_seconds=0, fallback={'fallback': True}, wait_seconds=0)
        assert state == 'refreshing'

        class ErrorFuture:
            def result(self, timeout=None):
                raise RuntimeError('loader failed')
        monkeypatch.setattr(cache, '_refresh', lambda key, loader: ErrorFuture())
        value, state = cache.get('err', lambda: {'x': 4}, ttl_seconds=0, fallback={'fallback': True})
        assert state == 'error'

        cache.invalidate('fresh')
        cache.invalidate()

        completed = Future()
        completed.set_result({'oldest': True})
        cache._store_completed('oldest', completed)
        newer = Future()
        newer.set_result({'newest': True})
        cache._store_completed('newest', newer)
        assert list(cache._entries) == ['newest']

        failed = Future()
        failed.set_exception(RuntimeError('background loader failed'))
        cache._in_flight['failed'] = failed
        cache._store_completed('failed', failed)
        assert 'failed' not in cache._in_flight

        def fail_write(self, *args, **kwargs):
            raise OSError('disk unavailable')

        with monkeypatch.context() as persist_patch:
            persist_patch.setattr(Path, 'write_text', fail_write)
            cache._persist()
    finally:
        cache.close()
