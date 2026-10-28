"""Custom rules (long only): build a strategy from switchable building blocks.

Every rule is a plain on/off switch plus a number, so a rule set is just a parameter set: it
can be saved, backtested, tuned out-of-sample and run live exactly like the hand-written
strategies, with the same no-lookahead guarantees. There is no code to paste in anywhere.

Entry: every ENABLED entry condition must be true on the same closed candle (AND).
  * trend      close is above its EMA(trend_len)
  * rsi        RSI(rsi_period) is below (dip) or above (momentum) rsi_level
  * breakout   close is above the highest high of the previous breakout_len candles
  * cross      fast EMA is above slow EMA (optionally: only on the candle it crosses)
Stop / target: stop = ATR * atr_stop_mult below the close, target = risk_reward times that.
Exit (any enabled one closes the trade at the next open, besides stop, target and time limit):
  * channel    close is below the lowest low of the previous exit_len candles
  * rsi        RSI(rsi_period) is above exit_rsi_level
"""
import numpy as np
import pandas as pd

from app.strategies.base import Signal, Strategy
from app.strategies.indicators import atr, ema, rsi


class Rules(Strategy):
    name = "rules"
    label = "Custom rules (build your own)"
    evidence = ("UNTESTED until you backtest it. Each entry condition you switch on makes signals rarer, and every "
                "parameter you tune is another chance to fit noise. Run the out-of-sample test before trusting one.")
    default_params = {
        "use_trend": 1, "trend_len": 100,
        "use_rsi": 1, "rsi_period": 14, "rsi_above": 0, "rsi_level": 40,
        "use_breakout": 0, "breakout_len": 40,
        "use_cross": 0, "ema_fast": 12, "ema_slow": 26, "cross_fresh": 0,
        "atr_period": 14, "atr_stop_mult": 2.0, "risk_reward": 2.0,
        "use_exit_channel": 0, "exit_len": 20,
        "use_exit_rsi": 0, "exit_rsi_level": 70,
    }
    flag_params = ("use_trend", "use_rsi", "rsi_above", "use_breakout", "use_cross", "cross_fresh",
                   "use_exit_channel", "use_exit_rsi")
    param_labels = {
        "use_trend": "Uptrend: close above the trend EMA", "trend_len": "Trend EMA length",
        "use_rsi": "RSI condition", "rsi_period": "RSI period",
        "rsi_above": "RSI must be ABOVE the level (unticked: below it, i.e. buy a dip)", "rsi_level": "RSI level",
        "use_breakout": "Breakout: close above the previous N-candle high", "breakout_len": "Breakout lookback N",
        "use_cross": "Fast EMA above slow EMA", "ema_fast": "Fast EMA length", "ema_slow": "Slow EMA length",
        "cross_fresh": "Only on the candle the cross happens",
        "atr_period": "ATR period", "atr_stop_mult": "Stop distance (× ATR)", "risk_reward": "Target (× the stop distance)",
        "use_exit_channel": "Exit when close drops below the previous M-candle low", "exit_len": "Exit lookback M",
        "use_exit_rsi": "Exit when RSI rises above a level", "exit_rsi_level": "Exit RSI level",
    }
    param_groups = [
        ("Entry: trend", ["use_trend", "trend_len"]),
        ("Entry: RSI", ["use_rsi", "rsi_period", "rsi_above", "rsi_level"]),
        ("Entry: breakout", ["use_breakout", "breakout_len"]),
        ("Entry: EMA cross", ["use_cross", "ema_fast", "ema_slow", "cross_fresh"]),
        ("Stop and target", ["atr_period", "atr_stop_mult", "risk_reward"]),
        ("Extra exits", ["use_exit_channel", "exit_len", "use_exit_rsi", "exit_rsi_level"]),
    ]
    param_grid = {"trend_len": [70, 100, 140], "rsi_level": [28, 40, 56], "atr_stop_mult": [2.0, 3.0]}

    ENTRY_FLAGS = ("use_trend", "use_rsi", "use_breakout", "use_cross")

    @staticmethod
    def _on(p, key) -> bool:
        return float(p[key]) >= 0.5

    def _p(self, params):
        return {**self.default_params, **params}

    def check(self, params):
        p = self._p(params)
        if not any(self._on(p, k) for k in self.ENTRY_FLAGS):
            return "Switch on at least one entry condition, otherwise this strategy can never enter"
        if self._on(p, "use_cross") and p["ema_fast"] >= p["ema_slow"]:
            return "The fast EMA must be shorter than the slow EMA"
        if min(p["trend_len"], p["rsi_period"], p["breakout_len"], p["ema_fast"], p["exit_len"], p["atr_period"]) < 2:
            return "Lengths and periods must be at least 2"
        if p["atr_stop_mult"] <= 0 or p["risk_reward"] <= 0:
            return "The stop distance and target must be greater than zero"
        return None

    def is_valid(self, params):
        return self.check(params) is None

    def grid_for(self, params):
        """Tune only the numbers belonging to rules that are switched on, at 0.7x / 1x / 1.4x (three
        rules at most), plus the stop distance. Keeps the number of tries, and so the luck, modest."""
        p = self._p(params)
        grid = {}
        for flag, key in (("use_trend", "trend_len"), ("use_breakout", "breakout_len"),
                          ("use_rsi", "rsi_level"), ("use_cross", "ema_slow")):
            if self._on(p, flag) and len(grid) < 3:
                v = p[key]
                vals = sorted({max(2, round(v * 0.7)), round(v), round(v * 1.4)})
                if key == "rsi_level":                       # a level, not a length: keep it inside 1..99
                    vals = [x for x in vals if 1 <= x <= 99] or [round(v)]
                grid[key] = vals
        grid["atr_stop_mult"] = sorted({p["atr_stop_mult"], p["atr_stop_mult"] + 1.0})
        return grid

    def describe(self, params):
        p = self._p(params)
        when = []
        if self._on(p, "use_trend"):
            when.append(f"close is above its EMA {int(p['trend_len'])}")
        if self._on(p, "use_rsi"):
            when.append(f"RSI({int(p['rsi_period'])}) is {'above' if self._on(p, 'rsi_above') else 'below'} {p['rsi_level']:g}")
        if self._on(p, "use_breakout"):
            when.append(f"close breaks the previous {int(p['breakout_len'])}-candle high")
        if self._on(p, "use_cross"):
            when.append(f"EMA {int(p['ema_fast'])} is above EMA {int(p['ema_slow'])}"
                        + (" (just crossed)" if self._on(p, "cross_fresh") else ""))
        text = ("Buy when " + " AND ".join(when)) if when else "No entry condition is switched on"
        text += f". Stop {p['atr_stop_mult']:g}×ATR({int(p['atr_period'])}) below, target {p['risk_reward']:g}R"
        outs = []
        if self._on(p, "use_exit_channel"):
            outs.append(f"close falls below the previous {int(p['exit_len'])}-candle low")
        if self._on(p, "use_exit_rsi"):
            outs.append(f"RSI rises above {p['exit_rsi_level']:g}")
        if outs:
            text += "; also exit when " + " or ".join(outs)
        return text + "."

    def plain(self, params):
        p = self._p(params)
        buy, terms, tags = [], ["stop", "target", "atr"], []
        if self._on(p, "use_trend"):
            buy.append(f"The coin is in an uptrend: its price is above its {int(p['trend_len'])}-candle average.")
            terms.append("ema"); tags.append("uptrend")
        if self._on(p, "use_rsi"):
            if self._on(p, "rsi_above"):
                buy.append(f"Momentum is strong: RSI, a 0-100 gauge of how stretched a move is, is above {p['rsi_level']:g}.")
                tags.append("strong momentum")
            else:
                buy.append(f"The coin has dipped: RSI, a 0-100 gauge of how stretched a move is, is below {p['rsi_level']:g}.")
                tags.append("dip")
            terms.append("rsi")
        if self._on(p, "use_breakout"):
            buy.append(f"The price closes above its highest price of the previous {int(p['breakout_len'])} candles (a breakout).")
            terms.append("breakout"); tags.append("breakout")
        if self._on(p, "use_cross"):
            buy.append(f"The quick average ({int(p['ema_fast'])} candles) is above the slow average ({int(p['ema_slow'])} candles)"
                       + (", and has only just crossed above it." if self._on(p, "cross_fresh") else ": recent prices are rising faster than the longer trend."))
            terms.append("ema"); tags.append("average cross")
        sell = [f"The price reaches the profit target ({p['risk_reward']:g}× the amount risked).",
                f"Or it falls to the stop-loss ({p['atr_stop_mult']:g}× the coin's typical candle swing below the entry)."]
        if self._on(p, "use_exit_channel"):
            sell.append(f"Or the price closes below the lowest price of the previous {int(p['exit_len'])} candles.")
        if self._on(p, "use_exit_rsi"):
            sell.append(f"Or RSI rises above {p['exit_rsi_level']:g}: the coin looks overheated.")
            terms.append("rsi")
        sell.append("Or the maximum holding time runs out.")
        return {
            "tagline": ("Buys on " + " + ".join(tags) + " (a custom rule set you built).") if tags else "A custom rule set with no entry condition switched on yet.",
            "idea": "A strategy built from the conditions you switched on. It only buys when every one of them is true on the same candle, "
                    "so each extra condition makes trades rarer and the rule stricter.",
            "buy": buy or ["No entry condition is switched on, so it can never buy."],
            "sell": sell,
            "watch": ["Every condition you add makes signals rarer, and every number you tune is another chance to fit the past by luck. "
                      "Trust it only after the out-of-sample test."],
            "terms": list(dict.fromkeys(terms)),
        }

    def warmup_bars(self, params):
        p = self._p(params)
        longest = 0
        if self._on(p, "use_trend"):
            longest = max(longest, 3 * int(p["trend_len"]))
        if self._on(p, "use_rsi") or self._on(p, "use_exit_rsi"):
            longest = max(longest, 5 * int(p["rsi_period"]))
        if self._on(p, "use_breakout"):
            longest = max(longest, int(p["breakout_len"]) + 20)
        if self._on(p, "use_cross"):
            longest = max(longest, 3 * int(p["ema_slow"]))
        if self._on(p, "use_exit_channel"):
            longest = max(longest, int(p["exit_len"]) + 20)
        return max(longest, 3 * int(p["atr_period"]), 100)

    def _conditions(self, df, p):
        """[(label, boolean array)] for every enabled entry condition, in a fixed order."""
        close = df["close"]
        out = []
        if self._on(p, "use_trend"):
            out.append((f"Close is above its EMA {int(p['trend_len'])} (uptrend)",
                        (close > ema(close, int(p["trend_len"]))).to_numpy()))
        if self._on(p, "use_rsi"):
            r = rsi(close, int(p["rsi_period"]))
            above = self._on(p, "rsi_above")
            out.append((f"RSI({int(p['rsi_period'])}) is {'above' if above else 'below'} {p['rsi_level']:g}",
                        ((r > p["rsi_level"]) if above else (r < p["rsi_level"])).to_numpy()))
        if self._on(p, "use_breakout"):
            # .shift(1): the PREVIOUS candles' range, so a candle can't be part of its own breakout level
            high_n = df["high"].rolling(int(p["breakout_len"])).max().shift(1)
            out.append((f"Close is above the highest high of the previous {int(p['breakout_len'])} candles",
                        (close > high_n).to_numpy()))
        if self._on(p, "use_cross"):
            fast, slow = ema(close, int(p["ema_fast"])), ema(close, int(p["ema_slow"]))
            up = fast > slow
            if self._on(p, "cross_fresh"):
                up = up & (fast.shift(1) <= slow.shift(1))
            out.append((f"EMA {int(p['ema_fast'])} is above EMA {int(p['ema_slow'])}"
                        + (" (crossed on this candle)" if self._on(p, "cross_fresh") else ""), up.to_numpy()))
        return out

    def context(self, df, params):
        p = self._p(params)
        if self._on(p, "use_trend"):
            return (df["close"] > ema(df["close"], int(p["trend_len"]))).to_numpy(copy=True)
        return np.ones(len(df), dtype=bool)

    def compute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        p = self._p(params)
        close = df["close"]
        a = atr(df, int(p["atr_period"]))
        enter = np.ones(len(df), dtype=bool)
        conds = self._conditions(df, p)
        for _, arr in conds:
            enter &= np.asarray(arr, dtype=bool)
        enter &= (a > 0).to_numpy()
        if not conds:
            enter[:] = False

        exit_ = np.zeros(len(df), dtype=bool)
        r = rsi(close, int(p["rsi_period"]))
        if self._on(p, "use_exit_channel"):
            exit_ |= (close < df["low"].rolling(int(p["exit_len"])).min().shift(1)).to_numpy()
        if self._on(p, "use_exit_rsi"):
            exit_ |= (r > p["exit_rsi_level"]).to_numpy()

        risk = a * p["atr_stop_mult"]
        d = pd.DataFrame(index=df.index)
        d["enter"], d["exit"] = enter, exit_
        d["stop"] = close - risk
        d["target"] = close + risk * p["risk_reward"]
        d["atr"] = a
        d["rsi"] = r if (self._on(p, "use_rsi") or self._on(p, "use_exit_rsi")) else float("nan")

        warm = self.warmup_bars(p)
        for col in ("enter", "exit"):
            d.iloc[: warm - 1, d.columns.get_loc(col)] = False
            d[col] = d[col].astype(bool)
        return d[["enter", "stop", "target", "exit", "atr", "rsi"]]

    def explain(self, df, params):
        p = self._p(params)
        warm = self.warmup_bars(p)
        settled = (pd.Series(range(len(df))) >= warm - 1).to_numpy()
        conds = self._conditions(df, p)
        if not conds:
            return []
        out = [(lab, arr) for lab, arr in conds]
        out[0] = (out[0][0], np.asarray(out[0][1], dtype=bool) & settled)
        out.append(("... and volatility (ATR) can be measured", (atr(df, int(p["atr_period"])) > 0).to_numpy()))
        return out

    def signal(self, symbol, df, params):
        if len(df) < self.warmup_bars(params):
            return None
        row = self.compute(df, params).iloc[-1]
        if not row["enter"]:
            return None
        return Signal(symbol=symbol, price=float(df["close"].iloc[-1]), stop=float(row["stop"]),
                      target=float(row["target"]), atr=float(row["atr"]), rsi=float(row["rsi"]),
                      candle_time=df["ts"].iloc[-1].isoformat())

    def exit_signal(self, df, params):
        if len(df) < self.warmup_bars(params):
            return False
        return bool(self.compute(df, params)["exit"].iloc[-1])
