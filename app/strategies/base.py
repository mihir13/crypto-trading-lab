"""The Strategy contract.

A strategy is a pure function of candles: give it closed candles, get back
signals. It knows nothing about balances, fees, scheduling or databases. That
separation is what lets the live bot, the backtester and any future strategy comparison all
run the same strategy code unchanged.

To add a strategy: subclass Strategy, fill in `default_params` (and optionally
`param_grid` for the Backtest Lab's tuning), implement `compute()`, then register
it in app/strategies/__init__.py.

`compute()` is the important one. It returns a signal for EVERY candle at once
(fast, and used by the backtester). `signal()` and `exit_signal()`, which the
live bot calls, just look at the last row of it, so live and backtest can never
disagree about what the strategy does.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Signal:
    symbol: str
    price: float          # price at the signal candle's close (reference only)
    stop: float
    target: float
    atr: float
    rsi: float
    candle_time: str      # ISO time of the CLOSED candle that produced the signal


class Strategy:
    name = "base"
    label = "Base"
    default_params: dict = {}
    # Values the Backtest Lab may try when tuning on the training period,
    # e.g. {"ema_fast": [5, 9, 12]}. Keep grids modest: every extra value
    # multiplies the number of tries, and more tries means more luck.
    param_grid: dict = {}
    # Plain-language note on how much evidence supports this strategy. Shown on the Futures Bot page
    # whenever the strategy is active, so nobody mistakes an experiment for a proven method.
    evidence: str = ""
    # Optional presentation hints for the strategy form (numbers-only params work without any):
    flag_params: tuple = ()          # params that are on/off switches (stored as 1/0), shown as checkboxes
    param_labels: dict = {}          # param -> human label
    param_groups: list = []          # [(heading, [param, ...]), ...]; empty = one flat list

    def compute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Signals for every candle in `df` (closed candles, oldest first).

        Returns a DataFrame with the same index and these columns:
          enter   bool   entry signal on this candle's CLOSE (trade happens next candle)
          stop    float  stop-loss level if entering at this candle's close
          target  float  take-profit level if entering at this candle's close
          exit    bool   close an open position at the next candle's open
          atr, rsi       for logging (NaN if the strategy doesn't use them)

        Row i may only use candles 0..i. Anything else is lookahead and makes
        backtests meaningless; tests/test_backtest.py checks this.

        This default is slow (calls signal() on every prefix) so any strategy
        works in the backtester; override it with a vectorised version for speed.
        """
        n = len(df)
        enter = np.zeros(n, dtype=bool)
        exit_ = np.zeros(n, dtype=bool)
        stop = np.full(n, np.nan)
        target = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        rsi = np.full(n, np.nan)
        for i in range(n):
            sub = df.iloc[: i + 1]
            sig = self.signal("", sub, params)
            if sig is not None:
                enter[i] = True
                stop[i], target[i], atr[i], rsi[i] = sig.stop, sig.target, sig.atr, sig.rsi
            exit_[i] = self.exit_signal(sub, params)
        return pd.DataFrame({"enter": enter, "stop": stop, "target": target, "exit": exit_,
                             "atr": atr, "rsi": rsi}, index=df.index)

    def context(self, df: pd.DataFrame, params: dict) -> np.ndarray:
        """Candles where the strategy is even willing to look for an entry (its regime filter).

        Default: every candle. A trend strategy would return "close is above its long EMA".
        The Backtest Lab uses this for a second random-entry baseline that only enters at
        random INSIDE the regime, which separates "the entry trigger adds value" from "just
        being in an uptrend does".
        """
        return np.ones(len(df), dtype=bool)

    def warmup_bars(self, params: dict) -> int:
        """Candles of history the indicators need before their values can be trusted.

        A long moving average (say 200 periods) is still dominated by its starting value
        after 200 candles and only settles after roughly three times its length. The
        backtester starts trading after this many candles, and the strategy stays silent
        until it has them.
        """
        return 60

    def grid_for(self, params: dict) -> dict:
        """The tuning grid to use for these parameters (default: the fixed `param_grid`).
        A strategy whose parameters switch whole rules on and off can tune only what is on."""
        return self.param_grid

    def check(self, params: dict) -> str | None:
        """A plain-language reason these parameters can't work, or None. Shown when saving."""
        return None

    def describe(self, params: dict) -> str:
        """One-sentence plain-English summary of what these parameters do (optional)."""
        return ""

    def plain(self, params: dict) -> dict:
        """The strategy in plain English for someone who doesn't know the jargon, using these
        parameter values. Keys: tagline (one line), idea (a short paragraph), buy (when it buys),
        sell (when it leaves), watch (what to be wary of), terms (glossary keys it mentions)."""
        return {"tagline": self.label, "idea": "", "buy": [], "sell": [], "watch": [], "terms": []}

    def is_valid(self, params: dict) -> bool:
        """Lets a strategy reject nonsense parameter combinations during tuning (e.g. fast EMA >= slow EMA)."""
        return True

    def explain(self, df: pd.DataFrame, params: dict) -> list | None:
        """Optional: the entry rule as an ordered list of (label, boolean array) conditions.

        Each condition is tested on its own; the Backtest Lab combines them into a
        "signal funnel" (how many candles pass condition 1, then 1+2, then 1+2+3 ...)
        so you can see which part of a rule is blocking trades.
        """
        return None

    def signal(self, symbol: str, df: pd.DataFrame, params: dict) -> Signal | None:
        """`df` contains CLOSED candles only; df.iloc[-1] is the latest closed one."""
        raise NotImplementedError

    def exit_signal(self, df: pd.DataFrame, params: dict) -> bool:
        """True if an open position should be closed at market now."""
        return False
