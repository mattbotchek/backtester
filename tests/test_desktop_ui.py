import importlib
import sys
import types
from unittest.mock import Mock, patch


class FakeWidget:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.children = []
        self.items = []
        self.text = ''
        self.value = ''

    def pack(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def insert(self, index, value):
        self.text += str(value)
        self.value = str(value)

    def delete(self, *args, **kwargs):
        self.text = ''
        self.items.clear()

    def get(self, *args, **kwargs):
        return self.value or self.text

    def set(self, value):
        self.value = value

    def config(self, **kwargs):
        self.kwargs.update(kwargs)

    configure = config

    def update_idletasks(self):
        return None

    def heading(self, *args, **kwargs):
        return None

    def column(self, *args, **kwargs):
        return None

    def add(self, frame, text=''):
        self.children.append((frame, text))

    def add_command(self, *args, **kwargs):
        return None

    def add_separator(self, *args, **kwargs):
        return None

    def add_cascade(self, *args, **kwargs):
        return None

    def get_children(self):
        return list(range(len(self.items)))

    def insert_tree(self, *args, **kwargs):
        self.items.append(kwargs.get('values'))

    def insert(self, *args, **kwargs):
        if kwargs.get('values') is not None:
            self.items.append(kwargs.get('values'))
        elif len(args) >= 2:
            self.text += str(args[1])
        elif args:
            self.text += str(args[-1])

    def after(self, delay, callback):
        callback()


class FakeRoot(FakeWidget):
    def title(self, value):
        self.title_value = value

    def geometry(self, value):
        self.geometry_value = value

    def minsize(self, *args):
        self.minsize_value = args

    def quit(self):
        self.quit_called = True

    def mainloop(self):
        self.mainloop_called = True


class FakeStyle:
    def theme_use(self, *args, **kwargs):
        return None

    def configure(self, *args, **kwargs):
        return None


class ImmediateThread:
    def __init__(self, target=None, daemon=None, name=None):
        self.target = target

    def start(self):
        if self.target:
            self.target()


class FakePrediction:
    def __init__(self):
        self.direction = 1
        self.predicted_return = 0.02
        self.confidence = 0.85
        self.outliers = ['gap']


class FakePredictor:
    def load(self):
        return None

    def predict_ticker(self, ticker):
        return FakePrediction()

    def train(self, tickers, timeline='5y'):
        return {'n_samples': 10, 'cv_accuracy_mean': 0.75}

    def save(self):
        return None


class FakeTrade:
    def __init__(self, pnl_pct):
        self.pnl_pct = pnl_pct


class FakeStrategyImpl:
    name = 'Fake Strategy'
    is_novel = True

    def run(self, data):
        return [FakeTrade(1.5), FakeTrade(-0.5)]


class FakeBacktestReport:
    pass


def _load_ui_module(monkeypatch):
    root = FakeRoot()
    ttk = types.SimpleNamespace(
        Style=FakeStyle,
        Frame=FakeWidget,
        LabelFrame=FakeWidget,
        Label=FakeWidget,
        Entry=FakeWidget,
        Combobox=FakeWidget,
        Button=FakeWidget,
        Treeview=FakeWidget,
        Notebook=FakeWidget,
    )
    messagebox = types.SimpleNamespace(
        showerror=Mock(),
        showinfo=Mock(),
    )
    scrolledtext = types.SimpleNamespace(ScrolledText=FakeWidget)
    tk_module = types.SimpleNamespace(
        Tk=lambda: root,
        Menu=FakeWidget,
        BOTH='both',
        X='x',
        BOTTOM='bottom',
        W='w',
        END='end',
        SUNKEN='sunken',
        ttk=ttk,
        messagebox=messagebox,
        scrolledtext=scrolledtext,
    )
    monkeypatch.setitem(sys.modules, 'tkinter', tk_module)
    monkeypatch.setitem(sys.modules, 'tkinter.ttk', ttk)
    monkeypatch.setitem(sys.modules, 'tkinter.messagebox', messagebox)
    monkeypatch.setitem(sys.modules, 'tkinter.scrolledtext', scrolledtext)
    sys.modules.pop('ui.desktop', None)
    module = importlib.import_module('ui.desktop')
    return module, root, messagebox


def test_desktop_workers_and_actions(monkeypatch):
    module, root, messagebox = _load_ui_module(monkeypatch)
    monkeypatch.setattr(module.threading, 'Thread', ImmediateThread)
    monkeypatch.setattr(module, 'messagebox', messagebox)
    monkeypatch.setitem(sys.modules, 'ml.model', types.SimpleNamespace(StockPredictor=FakePredictor))
    monkeypatch.setitem(sys.modules, 'strategies', types.SimpleNamespace(get_registry=lambda: {'fake': FakeStrategyImpl}))
    monkeypatch.setitem(
        sys.modules,
        'backtesting.engine',
        types.SimpleNamespace(
            BacktestConfig=lambda **kwargs: kwargs,
            run_backtest=lambda config: FakeBacktestReport(),
            print_report=lambda report: print('printed report'),
        ),
    )
    fake_history = types.SimpleNamespace(history=lambda **kwargs: __import__('pandas').DataFrame({'Close': [1, 2], 'Open': [1, 2], 'High': [1, 2], 'Low': [1, 2], 'Volume': [1, 1]}))
    monkeypatch.setitem(sys.modules, 'yfinance', types.SimpleNamespace(Ticker=lambda ticker: fake_history))

    ui = module.StockPredictorUI()
    ui.strat_ticker.value = 'aapl'
    ui.strat_timeline.value = '1y'
    ui.ml_ticker.value = 'msft'
    ui.bt_tickers.value = 'AAPL,MSFT'

    with patch('webbrowser.open') as open_browser:
        ui._run_strategies()
        ui._run_prediction()
        ui._train_model()
        ui._run_backtest()
        ui._open_web()
        ui._show_about()
        ui.run()

    assert ui.strat_tree.items
    assert ui.ml_direction.kwargs.get('text') == 'UP ^'
    assert 'Training started' in ui.ml_output.text
    assert 'printed report' in ui.bt_output.text
    open_browser.assert_called_once()
    messagebox.showinfo.assert_called()
    assert root.mainloop_called is True
