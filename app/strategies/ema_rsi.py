"""EMA crossover + RSI + volume filter (the original bot's strategy, long only)."""
import pandas as pd

from app.strategies.base import Signal, Strategy
from app.strategies.indicators import atr, ema, rsi

MIN_CANDLES = 60      # indicators need this much history to settle


class EmaRsi(Strategy):
    name = "ema_rsi"
    label = "EMA cross + RSI + volume"
    evidence = ("EXPERIMENT. An EMA cross-up with RSI and volume filters. Its default conditions can conflict with each "
                "other, so it may signal rarely or never until loosened; judge it with the lab's backtests.")
    default_params = {
        "ema_fast": 9, "ema_slow": 21,
        "rsi_period": 14, "rsi_entry_max": 50, "rsi_exit": 70,
        "atr_period": 14, "atr_stop_mult": 1.8, "risk_reward": 2.5,
        "vol_mult": 1.2,
    }
    param_grid = {
        "ema_fast": [5, 9, 12],
        "ema_slow": [21, 34, 50],
        "rsi_entry_max": [50, 60, 70],
        "vol_mult": [0.0, 1.2],
        "atr_stop_mult": [1.5, 1.8, 2.5],
        "risk_reward": [1.5, 2.0, 2.5, 3.0],
    }

    def compute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        p = {**self.default_params, **params}
        d = pd.DataFrame(index=df.index)
        close = df["close"]
        fast = ema(close, int(p["ema_fast"]))
        slow = ema(close, int(p["ema_slow"]))
        d["rsi"] = rsi(close, int(p["rsi_period"]))
        d["atr"] = atr(df, int(p["atr_period"]))
        vol_ma = df["vol"].rolling(20).mean()

        above = fast > slow
        cross_up = above & ~above.shift(fill_value=False)
        # Comparisons with NaN are False, so warm-up rows can never signal.
        d["enter"] = (cross_up
                      & (d["rsi"] < p["rsi_entry_max"])
                      & (df["vol"] >= vol_ma * p["vol_mult"])
                      & (d["atr"] > 0))
        d["exit"] = d["rsi"] > p["rsi_exit"]

        risk = d["atr"] * p["atr_stop_mult"]
        d["stop"] = close - risk
        d["target"] = close + risk * p["risk_reward"]

        # Same rule the live bot always had: no signals before MIN_CANDLES of history.
        d.iloc[: MIN_CANDLES - 1, d.columns.get_loc("enter")] = False
        d.iloc[: MIN_CANDLES - 1, d.columns.get_loc("exit")] = False
        d["enter"] = d["enter"].astype(bool)
        d["exit"] = d["exit"].astype(bool)
        return d[["enter", "stop", "target", "exit", "atr", "rsi"]]

    def plain(self, params):
        p = {**self.default_params, **params}
        buy = [f"The quick average (last {int(p['ema_fast'])} candles) crosses above the slow average (last {int(p['ema_slow'])} candles): "
               "recent prices are rising faster than the longer trend.",
               f"The coin isn't already overheated: its RSI is below {p['rsi_entry_max']:g}."]
        if p["vol_mult"] > 0:
            buy.append(f"Trading volume is at least {p['vol_mult']:g}× its recent average, so real interest is behind the move.")
        return {
            "tagline": "Buys when short-term momentum flips upward, if the coin isn't overheated and volume backs it.",
            "idea": "It tracks two averages of the price, a quick one and a slow one. When the quick one climbs above the slow one, "
                    "the price has started rising faster than its longer-term trend, which can be the start of an upswing. "
                    "It buys that moment, as long as the coin hasn't already run up too far.",
            "buy": buy,
            "sell": [f"The price reaches the profit target ({p['risk_reward']:g}× the amount risked).",
                     f"Or it falls to the stop-loss ({p['atr_stop_mult']:g}× the coin's typical candle swing below the entry).",
                     f"Or the coin becomes overheated (RSI above {p['rsi_exit']:g}).",
                     "Or the maximum holding time runs out."],
            "watch": ["Averages react late: part of the move has usually happened by the time they cross.",
                      "In choppy, sideways markets crossovers fire repeatedly and mostly lose small amounts."],
            "terms": ["ema", "rsi", "atr", "volume", "stop", "target"],
        }

    def warmup_bars(self, params):
        return MIN_CANDLES

    def is_valid(self, params):
        p = {**self.default_params, **params}
        return p["ema_fast"] < p["ema_slow"]

    def explain(self, df, params):
        p = {**self.default_params, **params}
        close = df["close"]
        above = ema(close, int(p["ema_fast"])) > ema(close, int(p["ema_slow"]))
        cross_up = (above & ~above.shift(fill_value=False)).to_numpy(copy=True)
        cross_up[: MIN_CANDLES - 1] = False
        r = rsi(close, int(p["rsi_period"]))
        vol_ok = (df["vol"] >= df["vol"].rolling(20).mean() * p["vol_mult"]).to_numpy()
        return [
            (f"EMA {int(p['ema_fast'])} crosses above EMA {int(p['ema_slow'])}", cross_up),
            (f"... and RSI is below {p['rsi_entry_max']:g}", (r < p["rsi_entry_max"]).to_numpy()),
            (f"... and volume is at least {p['vol_mult']:g}x its 20-candle average", vol_ok),
        ]

    def signal(self, symbol, df, params):
        if len(df) < MIN_CANDLES:
            return None
        row = self.compute(df, params).iloc[-1]
        if not row["enter"]:
            return None
        return Signal(
            symbol=symbol, price=float(df["close"].iloc[-1]),
            stop=float(row["stop"]), target=float(row["target"]),
            atr=float(row["atr"]), rsi=float(row["rsi"]),
            candle_time=df["ts"].iloc[-1].isoformat(),
        )

    def exit_signal(self, df, params):
        if len(df) < MIN_CANDLES:
            return False
        return bool(self.compute(df, params)["exit"].iloc[-1])
