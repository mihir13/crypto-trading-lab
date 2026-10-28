"""Trend pullback (long only): buy a dip that happens inside an uptrend.

The idea: in an uptrend, short-lived dips tend to be followed by a resumption, so
buying the moment momentum turns back up gets a better price than chasing strength.
The trend filter (close above its 200-period EMA) is what keeps it mostly out of
downtrends, which is where the original EMA/RSI rule and long-only trading in general
struggled.

Parameters fixed in docs/futures-strategy-protocol.md before any test was run.
"""
import pandas as pd

from app.strategies.base import Signal, Strategy
from app.strategies.indicators import atr, ema, rsi


class TrendPullback(Strategy):
    name = "trend_pullback"
    label = "Trend pullback (buy dips above EMA 200)"
    evidence = ("EXPERIMENT. Buys RSI dips while price is above its 200-period EMA. Its parameters were fixed before "
                "testing; judge it with the lab's backtests and a paper bot, not by this label.")
    default_params = {
        "trend_len": 200,          # regime filter: only trade while close > EMA(trend_len)
        "rsi_period": 14,
        "rsi_buy": 40,             # buy when RSI crosses back UP through this after being below it
        "rsi_exit": 65,            # take profit early when RSI gets this high
        "atr_period": 14,
        "atr_stop_mult": 2.0,
        "risk_reward": 3.0,
    }
    param_grid = {
        "trend_len": [100, 200],
        "rsi_buy": [30, 40],
        "rsi_exit": [60, 70],
        "atr_stop_mult": [2.0, 3.0],
        "risk_reward": [2.0, 3.0],
    }

    def plain(self, params):
        p = {**self.default_params, **params}
        return {
            "tagline": "Buys a short dip inside an uptrend, betting the trend resumes.",
            "idea": f"It first checks that the coin is in an uptrend (price above its {int(p['trend_len'])}-candle average). "
                    "Then it waits for a dip and buys the moment the price starts recovering, rather than chasing a rally at a higher price.",
            "buy": [f"The price is above its {int(p['trend_len'])}-candle average, so the coin is in an uptrend.",
                    f"RSI, a 0-100 gauge of how stretched a move is, was below {p['rsi_buy']:g} (a dip) and has just climbed back up through {p['rsi_buy']:g}."],
            "sell": [f"The price reaches the profit target ({p['risk_reward']:g}× the amount risked).",
                     f"Or it falls to the stop-loss ({p['atr_stop_mult']:g}× the coin's typical candle swing below the entry).",
                     f"Or RSI rises above {p['rsi_exit']:g}: take the profit while the coin is strong.",
                     "Or the uptrend breaks (price falls below its average).",
                     "Or the maximum holding time runs out."],
            "watch": ["A dip in an uptrend can also be the start of a reversal. The stop-loss caps each loss, but it needs winners to outweigh the small losses.",
                      "Few trades happen while the coin is in a downtrend, so it can sit idle for a long time."],
            "terms": ["ema", "rsi", "atr", "stop", "target"],
        }

    def warmup_bars(self, params):
        p = {**self.default_params, **params}
        return max(3 * int(p["trend_len"]), 100)

    def context(self, df, params):
        p = {**self.default_params, **params}
        return (df["close"] > ema(df["close"], int(p["trend_len"]))).to_numpy(copy=True)

    def compute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        p = {**self.default_params, **params}
        close = df["close"]
        trend = ema(close, int(p["trend_len"]))
        r = rsi(close, int(p["rsi_period"]))
        a = atr(df, int(p["atr_period"]))

        in_uptrend = close > trend
        turned_up = (r.shift(1) < p["rsi_buy"]) & (r >= p["rsi_buy"])

        d = pd.DataFrame(index=df.index)
        d["enter"] = in_uptrend & turned_up & (a > 0)
        # leave on strength, or as soon as the regime that justified the trade is gone
        d["exit"] = (r > p["rsi_exit"]) | ~in_uptrend
        risk = a * p["atr_stop_mult"]
        d["stop"] = close - risk
        d["target"] = close + risk * p["risk_reward"]
        d["atr"], d["rsi"] = a, r

        warm = self.warmup_bars(p)
        for col in ("enter", "exit"):
            d.iloc[: warm - 1, d.columns.get_loc(col)] = False
            d[col] = d[col].astype(bool)
        return d[["enter", "stop", "target", "exit", "atr", "rsi"]]

    def explain(self, df, params):
        p = {**self.default_params, **params}
        close = df["close"]
        r = rsi(close, int(p["rsi_period"]))
        warm = self.warmup_bars(p)
        settled = (pd.Series(range(len(df))) >= warm - 1).to_numpy()
        return [
            (f"Close is above its EMA {int(p['trend_len'])} (uptrend)", ((close > ema(close, int(p['trend_len']))).to_numpy() & settled)),
            (f"... and RSI was below {p['rsi_buy']:g} on the previous candle", (r.shift(1) < p["rsi_buy"]).to_numpy()),
            (f"... and RSI is now back at or above {p['rsi_buy']:g}", (r >= p["rsi_buy"]).to_numpy()),
        ]

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
