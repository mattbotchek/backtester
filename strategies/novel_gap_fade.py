"""[NOVEL] Gap Fade strategy.

EXPERIMENTAL — This strategy is NOT based on any known published strategy.

Concept: When a stock gaps up >1.5% at open vs previous close, the gap
tends to "fill" (fade back). This strategy shorts gaps up by selling at open
and rebuying later. Since we're long-only, we instead avoid buying gap-ups
and buy gap-downs expecting a bounce, selling when the gap fills or EOD.
"""

import math
from .strategy import Strategy, TradeResult


class GapFade(Strategy):
    name = "GapFade"
    description = "[NOVEL] Buys gap-downs expecting fill; avoids gap-ups."
    is_novel = True

    def __init__(self, timeline, interval, ticker, cash, gap_threshold=0.015):
        super().__init__(timeline, interval, ticker, cash)
        self.gap_threshold = gap_threshold

    def calculate(self, verbose: bool = False) -> TradeResult:
        df = self.history.copy()

        in_position = False
        prev_close = None
        gap_target = None

        for date, row in df.iterrows():
            price_open = float(row["Open"])
            price_close = float(row["Close"])

            if prev_close is not None:
                gap_pct = (price_open - prev_close) / prev_close

                # Gap down: buy expecting fill
                if gap_pct < -self.gap_threshold and not in_position:
                    self._buy(price_open, date=date)
                    in_position = True
                    gap_target = prev_close  # target = gap fill
                    if verbose:
                        print(f"  BUY  @ {price_open:.2f} (gap down {gap_pct:.2%})")

            if in_position and gap_target is not None:
                if price_close >= gap_target:
                    self._sell(price_close, date=date)
                    in_position = False
                    if verbose:
                        print(f"  SELL @ {price_close:.2f} (gap filled)")
                elif price_close < price_open * 0.97:
                    # Stop loss if gap doesn't fill and continues dropping
                    self._sell(price_close, date=date)
                    in_position = False
                    if verbose:
                        print(f"  SELL @ {price_close:.2f} (stop loss)")

            prev_close = price_close

        if self.shares > 0:
            self._sell(float(df.iloc[-1]["Close"]), date=df.index[-1])

        return self._calculate_metrics()
