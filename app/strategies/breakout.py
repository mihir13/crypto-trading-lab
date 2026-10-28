"""Breakout (long only): buy a new high inside an uptrend, leave when a recent low breaks.

The idea: trends persist, so a close above the highest high of the last N candles is a
sign the move has real buying behind it. There is no profit target (it is set 10R away,
effectively never); the trade ends when price closes below the lowest low of the last M
candles, or on the ATR stop, or the time limit.

Parameters fixed in docs/futures-strategy-protocol.md before any test was run.
"""
import pandas as pd

from app.strategies.base import Signal, Strategy
from app.strategies.indicators import atr, ema


class Breakout(Strategy):
    name = "breakout"
    label = "Breakout (new highs above EMA 200)"
    evidence = ("EXPERIMENT. Buys new highs while price is above its 200-period EMA. Its parameters were fixed before "
                "testing; judge it with the lab's backtests and a paper bot, not by this label.")
    default_params = {
        "trend_len": 200,          # regime filter: only trade while close > EMA(trend_len)
        "breakout_len": 40,        # enter when close is above the highest high of the previous N candles
        "exit_len": 20,            # exit when close is below the lowest low of the previous M candles
        "atr_period": 14,
        "atr_stop_mult": 3.0,
        "risk_reward": 10.0,       # effectively no target; the exit rule ends the trade
    }
    param_grid = {
        "breakout_len": [20, 40, 60],
        "exit_len": [10, 20],
        "trend_len": [100, 200],
        "atr_stop_mult": [2.0, 3.0],
    }

    def plain(self, params):
        p = {**self.default_params, **params}
        no_target = p["risk_reward"] >= 8
        return {
            "tagline": "Buys when price breaks to a new high inside an uptrend, and rides it until the move fades.",
            "idea": "Trends tend to keep going. When a coin in an uptrend closes above its highest price of the last few weeks, "
                    "it suggests real buying pressure. The strategy jumps in and stays in the trade until the price starts making new lows again.",
            "buy": [f"The price is above its {int(p['trend_len'])}-candle average, so the coin is in an uptrend.",
                    f"It closes above the highest price of the previous {int(p['breakout_len'])} candles (a breakout)."],
            "sell": [f"The price closes below the lowest price of the previous {int(p['exit_len'])} candles: the move is fading.",
                     f"Or it falls to the stop-loss ({p['atr_stop_mult']:g}× the coin's typical candle swing below the entry).",
                     "There is effectively no profit target, so winning trades are allowed to run." if no_target
                     else f"Or it reaches the profit target ({p['risk_reward']:g}× the amount risked).",
                     "Or the maximum holding time runs out."],
            "watch": ["Many breakouts fail and reverse. Expect a lot of small losses, with a few big winners providing the profit.",
                      "Long stretches without profit are normal for this style."],
            "terms": ["ema", "breakout", "atr", "stop", "target"],
        }

    def warmup_bars(self, params):
        p = {**self.default_params, **params}
        return max(3 * int(p["trend_len"]), int(p["breakout_len"]) + 20, 100)

    def is_valid(self, params):
        p = {**self.default_params, **params}
        return p["exit_len"] <= p["breakout_len"]

    def context(self, df, params):
        p = {**self.default_params, **params}
        return (df["close"] > ema(df["close"], int(p["trend_len"]))).to_numpy(copy=True)

    def compute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        p = {**self.default_params, **params}
        close = df["close"]
        trend = ema(close, int(p["trend_len"]))
        a = atr(df, int(p["atr_period"]))
        # .shift(1): the range of the PREVIOUS candles, so the current candle can't be part of its own breakout level
        high_n = df["high"].rolling(int(p["breakout_len"])).max().shift(1)
        low_m = df["low"].rolling(int(p["exit_len"])).min().shift(1)

        d = pd.DataFrame(index=df.index)
        d["enter"] = (close > trend) & (close > high_n) & (a > 0)
        d["exit"] = close < low_m
        risk = a * p["atr_stop_mult"]
        d["stop"] = close - risk
        d["target"] = close + risk * p["risk_reward"]
        d["atr"], d["rsi"] = a, float("nan")

        warm = self.warmup_bars(p)
        for col in ("enter", "exit"):
            d.iloc[: warm - 1, d.columns.get_loc(col)] = False
            d[col] = d[col].astype(bool)
        return d[["enter", "stop", "target", "exit", "atr", "rsi"]]

    def explain(self, df, params):
        p = {**self.default_params, **params}
        close = df["close"]
        warm = self.warmup_bars(p)
        settled = (pd.Series(range(len(df))) >= warm - 1).to_numpy()
        high_n = df["high"].rolling(int(p["breakout_len"])).max().shift(1)
        return [
            (f"Close is above its EMA {int(p['trend_len'])} (uptrend)", ((close > ema(close, int(p['trend_len']))).to_numpy() & settled)),
            (f"... and close is above the highest high of the previous {int(p['breakout_len'])} candles", (close > high_n).to_numpy()),
        ]

    def signal(self, symbol, df, params):
        if len(df) < self.warmup_bars(params):
            return None
        row = self.compute(df, params).iloc[-1]
        if not row["enter"]:
            return None
        return Signal(symbol=symbol, price=float(df["close"].iloc[-1]), stop=float(row["stop"]),
                      target=float(row["target"]), atr=float(row["atr"]), rsi=float("nan"),
                      candle_time=df["ts"].iloc[-1].isoformat())

    def exit_signal(self, df, params):
        if len(df) < self.warmup_bars(params):
            return False
        return bool(self.compute(df, params)["exit"].iloc[-1])
