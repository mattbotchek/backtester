"""Local desktop UI for Stock Predictor.

Tkinter-based GUI with:
  - Strategy controls and results
  - ML model training/prediction
  - Ticker search and selection
  - Headless mode toggle

Run: python -m ui.desktop
"""

import logging
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class StockPredictorUI:
    """Desktop UI for the stock predictor system."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stock Predictor")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        self._create_styles()
        self._create_menu()
        self._create_notebook()
        self._create_status_bar()

    def _create_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook.Tab", padding=[12, 4])
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Web UI", command=self._open_web)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _create_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self._create_strategy_tab()
        self._create_ml_tab()
        self._create_backtest_tab()
        self._create_log_tab()
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _create_strategy_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Strategy Analysis")

        ctrl = ttk.LabelFrame(frame, text="Controls")
        ctrl.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(ctrl, text="Ticker:").grid(row=0, column=0, padx=4, pady=4)
        self.strat_ticker = ttk.Entry(ctrl, width=12)
        self.strat_ticker.insert(0, "AAPL")
        self.strat_ticker.grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(ctrl, text="Timeline:").grid(row=0, column=2, padx=4, pady=4)
        self.strat_timeline = ttk.Combobox(ctrl, values=["3mo", "6mo", "1y", "2y", "5y"], width=6)
        self.strat_timeline.set("1y")
        self.strat_timeline.grid(row=0, column=3, padx=4, pady=4)

        ttk.Button(ctrl, text="Run All Strategies", command=self._run_strategies).grid(row=0, column=4, padx=8)

        cols = ("name", "return", "winrate", "trades", "type")
        self.strat_tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
        self.strat_tree.heading("name", text="Strategy")
        self.strat_tree.heading("return", text="Return %")
        self.strat_tree.heading("winrate", text="Win Rate %")
        self.strat_tree.heading("trades", text="Trades")
        self.strat_tree.heading("type", text="Type")
        self.strat_tree.column("name", width=220)
        self.strat_tree.column("return", width=100)
        self.strat_tree.column("winrate", width=100)
        self.strat_tree.column("trades", width=80)
        self.strat_tree.column("type", width=80)
        self.strat_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _create_ml_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ML Prediction")

        ctrl = ttk.LabelFrame(frame, text="Prediction Controls")
        ctrl.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(ctrl, text="Ticker:").grid(row=0, column=0, padx=4, pady=4)
        self.ml_ticker = ttk.Entry(ctrl, width=12)
        self.ml_ticker.insert(0, "AAPL")
        self.ml_ticker.grid(row=0, column=1, padx=4, pady=4)

        ttk.Button(ctrl, text="Predict", command=self._run_prediction).grid(row=0, column=2, padx=8)
        ttk.Button(ctrl, text="Train Model", command=self._train_model).grid(row=0, column=3, padx=8)

        result_frame = ttk.LabelFrame(frame, text="Result")
        result_frame.pack(fill=tk.X, padx=8, pady=4)

        self.ml_direction = ttk.Label(result_frame, text="--", font=("Segoe UI", 18, "bold"))
        self.ml_direction.pack(pady=8)
        self.ml_details = ttk.Label(result_frame, text="Enter a ticker and click Predict")
        self.ml_details.pack(pady=4)

        self.ml_output = scrolledtext.ScrolledText(frame, height=15, font=("Consolas", 9))
        self.ml_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _create_backtest_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Backtest")

        ctrl = ttk.LabelFrame(frame, text="Controls")
        ctrl.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(ctrl, text="Tickers:").grid(row=0, column=0, padx=4, pady=4)
        self.bt_tickers = ttk.Entry(ctrl, width=40)
        self.bt_tickers.insert(0, "AAPL,MSFT,GOOGL,AMZN,JPM")
        self.bt_tickers.grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(ctrl, text="Run Backtest", command=self._run_backtest).grid(row=0, column=2, padx=8)

        self.bt_output = scrolledtext.ScrolledText(frame, height=25, font=("Consolas", 9))
        self.bt_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _create_log_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Log")
        self.log_output = scrolledtext.ScrolledText(frame, height=30, font=("Consolas", 9))
        self.log_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _create_status_bar(self):
        self.status = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _set_status(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()

    def _run_strategies(self):
        ticker = self.strat_ticker.get().upper()
        timeline = self.strat_timeline.get()
        self._set_status(f"Running strategies on {ticker}...")

        def worker():
            try:
                import yfinance
                from strategies import get_registry

                data = yfinance.Ticker(ticker).history(period=timeline)
                if data.empty:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"No data for {ticker}"))
                    return

                strategies = list(get_registry().values())
                results = []

                for strategy_cls in strategies:
                    try:
                        strategy = strategy_cls()
                        trades = strategy.run(data)
                        if trades:
                            total_ret = sum(t.pnl_pct for t in trades)
                            win_rate = sum(1 for t in trades if t.pnl_pct > 0) / len(trades) * 100
                            results.append((
                                strategy.name, f"{total_ret:.2f}",
                                f"{win_rate:.1f}", str(len(trades)),
                                "NOVEL" if getattr(strategy, "is_novel", False) else "KNOWN",
                            ))
                    except Exception:
                        pass

                results.sort(key=lambda x: float(x[1]), reverse=True)

                def update_ui():
                    for item in self.strat_tree.get_children():
                        self.strat_tree.delete(item)
                    for name, ret, wr, trades, stype in results:
                        self.strat_tree.insert("", tk.END, values=(name, ret, wr, trades, stype))
                    self._set_status(f"Done: {len(results)} strategies on {ticker}")

                self.root.after(0, update_ui)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _run_prediction(self):
        ticker = self.ml_ticker.get().upper()
        self._set_status(f"Predicting {ticker}...")

        def worker():
            try:
                from ml.model import StockPredictor
                predictor = StockPredictor()
                predictor.load()
                pred = predictor.predict_ticker(ticker)

                dir_text = "UP ^" if pred.direction == 1 else ("DOWN v" if pred.direction == -1 else "FLAT -")
                color = "green" if pred.direction == 1 else ("red" if pred.direction == -1 else "black")

                def update():
                    self.ml_direction.config(text=dir_text, foreground=color)
                    self.ml_details.config(
                        text=f"Return: {pred.predicted_return * 100:.2f}%  |  "
                             f"Confidence: {pred.confidence * 100:.1f}%  |  "
                             f"Outliers: {', '.join(pred.outliers) if pred.outliers else 'None'}"
                    )
                    self._set_status(f"Prediction: {ticker} {dir_text}")

                self.root.after(0, update)
            except FileNotFoundError:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No trained model. Click 'Train Model' first."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _train_model(self):
        self._set_status("Training model...")
        self.ml_output.delete("1.0", tk.END)
        self.ml_output.insert(tk.END, "Training started...\n")

        def worker():
            try:
                from ml.model import StockPredictor
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
                           "JNJ", "V", "PG", "UNH", "HD", "DIS", "MA", "NFLX",
                           "XOM", "CVX", "WMT", "BA", "GS", "BAC", "PFE", "AMD"]

                predictor = StockPredictor()
                stats = predictor.train(tickers, timeline="5y")

                def update():
                    self.ml_output.insert(tk.END, f"Complete!\n")
                    self.ml_output.insert(tk.END, f"  Samples: {stats['n_samples']}\n")
                    self.ml_output.insert(tk.END, f"  CV Accuracy: {stats['cv_accuracy_mean']:.3f}\n")
                    self._set_status("Model trained")

                predictor.save()
                self.root.after(0, update)
            except Exception as e:
                self.root.after(0, lambda: self.ml_output.insert(tk.END, f"Error: {e}\n"))

        threading.Thread(target=worker, daemon=True).start()

    def _run_backtest(self):
        tickers = [t.strip() for t in self.bt_tickers.get().split(",")]
        self._set_status("Running backtest...")
        self.bt_output.delete("1.0", tk.END)

        def worker():
            try:
                from backtesting.engine import BacktestConfig, run_backtest, print_report
                import io

                config = BacktestConfig(tickers=tickers, timeline="1y", starting_cash=10000)
                report = run_backtest(config)

                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                print_report(report)
                sys.stdout = old_stdout
                output = buf.getvalue()

                self.root.after(0, lambda: self.bt_output.insert(tk.END, output))
                self.root.after(0, lambda: self._set_status("Backtest complete"))
            except Exception as e:
                sys.stdout = sys.__stdout__
                self.root.after(0, lambda: self.bt_output.insert(tk.END, f"Error: {e}\n"))

        threading.Thread(target=worker, daemon=True).start()

    def _open_web(self):
        import webbrowser
        webbrowser.open("http://localhost:8000")

    def _show_about(self):
        messagebox.showinfo("About", "Stock Predictor v1.0\n22 strategies + ML prediction\nFor educational use only.")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    StockPredictorUI().run()
