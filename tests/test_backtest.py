"""Tests for the backtest engine. Run:  python -m unittest discover -s tests -v

Each engine test builds a tiny synthetic market where the right answer can be worked
out by hand, then checks the engine agrees. If these fail, no backtest number can be
trusted, so they are the first thing to run after any change to app/backtest/.
"""
import unittest

import numpy as np
import pandas as pd

from app.backtest.engine import BtConfig, Market, simulate
from app.backtest import metrics
from app.strategies.base import Signal, Strategy
from app.strategies import STRATEGIES
from app.strategies.ema_rsi import EmaRsi

TS0 = 1_704_067_200_000            # 2024-01-01 00:00 UTC
TF = 4 * 3_600_000                 # 4h candles


def candles(n=40, price=100.0, over=None):
    """Flat candles at `price`; `over` = {index: (open, high, low, close)} to override."""
    df = pd.DataFrame({"ts_ms": TS0 + np.arange(n) * TF, "open": price, "high": price,
                       "low": price, "close": price, "vol": 1.0})
    for i, (o, h, l, c) in (over or {}).items():
        df.loc[i, ["open", "high", "low", "close"]] = [o, h, l, c]
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df


def sig(n, enters=None, exits=(), ):
    """Hand-written signals. enters = {candle_index: (stop, target)}: 'signal on the CLOSE of that candle'."""
    E, X = np.zeros(n, bool), np.zeros(n, bool)
    S, G = np.full(n, np.nan), np.full(n, np.nan)
    for i, (stop, target) in (enters or {}).items():
        E[i], S[i], G[i] = True, stop, target
    for i in exits:
        X[i] = True
    return {"E": E, "X": X, "S": S, "G": G}


def cfg(**kw):
    base = dict(capital=1000.0, risk_pct=1.0, leverage=5.0, max_margin_pct=100.0, max_simultaneous=3,
                taker_fee=0.001, slippage_pct=0.1, max_hold_hours=72.0, max_funding_abs=1.0,
                fallback_funding=0.0)
    base.update(kw)
    return BtConfig(**base)


def run(df, signals, config, funding=None, start=None, end=None, sym="AAA"):
    m = Market({sym: df}, {sym: funding} if funding else {}, TF)
    return simulate(m, {sym: signals}, config, start if start is not None else int(df.ts_ms.iloc[1]),
                    end if end is not None else int(df.ts_ms.iloc[-1]) + TF), m


class EngineRules(unittest.TestCase):
    def setUp(self):
        self.c = cfg()
        # Entry math for a signal on candle 10 (stop 95, target 110) when candle 11 opens at 100:
        self.fill = 100 * (1 + 0.001)                     # slippage paid on entry
        self.units = (1000 * 0.01) / (self.fill - 95)      # risk 1% of 1000 = 10, per-unit risk = fill - stop
        self.fee_open = self.units * self.fill * 0.001

    def test_take_profit_exact_pnl(self):
        df = candles(over={12: (100, 111, 99, 105)})       # high 111 reaches target 110, low stays above stop
        r, _ = run(df, sig(40, {10: (95, 110)}), self.c)
        self.assertEqual(len(r.trades), 1)
        t = r.trades[0]
        self.assertEqual(t["outcome"], "TAKE PROFIT")
        self.assertEqual(t["entry_ts"], TS0 + 11 * TF, "entry must be the candle AFTER the signal")
        self.assertAlmostEqual(t["entry"], self.fill)
        self.assertAlmostEqual(t["exit"], 110.0)           # target credited at the target, not better
        gross = (110 - self.fill) * self.units
        fee_close = 110 * self.units * 0.001
        self.assertAlmostEqual(t["net"], gross - self.fee_open - fee_close, places=9)
        self.assertAlmostEqual(r.final_balance, 1000 + t["net"], places=9)

    def test_stop_wins_when_candle_touches_both(self):
        df = candles(over={12: (100, 111, 94, 100)})       # low 94 <= stop 95 AND high 111 >= target 110
        r, _ = run(df, sig(40, {10: (95, 110)}), self.c)
        t = r.trades[0]
        self.assertEqual(t["outcome"], "STOP LOSS")
        self.assertAlmostEqual(t["exit"], 95 * 0.999)      # stop, minus slippage

    def test_gap_through_stop_fills_at_open(self):
        df = candles(over={12: (90, 91, 88, 90)})          # opens below the stop
        r, _ = run(df, sig(40, {10: (95, 110)}), self.c)
        t = r.trades[0]
        self.assertEqual(t["outcome"], "STOP LOSS")
        self.assertAlmostEqual(t["exit"], 90 * 0.999)      # NOT 95: a gap can't fill at the stop

    def test_full_stop_loses_about_one_r(self):
        df = candles(over={12: (100, 100, 94, 96)})
        r, _ = run(df, sig(40, {10: (95, 110)}), self.c)
        t = r.trades[0]
        self.assertAlmostEqual(t["risk_amount"], 10.0)
        self.assertTrue(-1.35 < t["r"] < -1.0, f"a stop-out should cost ~1R plus costs, got {t['r']:.3f}R")

    def test_time_stop_exits_at_next_open(self):
        df = candles(over={13: (102, 102, 102, 102)})       # entered at open of 11; 8h later = open of 13
        r, _ = run(df, sig(40, {10: (95, 110)}), cfg(max_hold_hours=8))
        t = r.trades[0]
        self.assertEqual(t["outcome"], "MAX HOLD")
        self.assertEqual(t["exit_ts"], TS0 + 13 * TF)
        self.assertAlmostEqual(t["exit"], 102 * 0.999)

    def test_strategy_exit_fills_at_next_open(self):
        df = candles(over={13: (103, 103, 103, 103)})
        r, _ = run(df, sig(40, {10: (95, 110)}, exits=[12]), self.c)    # exit flag on close of 12
        t = r.trades[0]
        self.assertEqual(t["outcome"], "STRATEGY EXIT")
        self.assertEqual(t["exit_ts"], TS0 + 13 * TF)
        self.assertAlmostEqual(t["exit"], 103 * 0.999)

    def test_no_lookahead_signal_uses_only_earlier_candles(self):
        # A huge move happens INSIDE candle 10; the signal on its close can only be acted on at candle 11.
        df = candles(over={10: (100, 130, 100, 130), 11: (130, 130, 130, 130)})
        r, _ = run(df, sig(40, {10: (95, 200)}), self.c)
        self.assertAlmostEqual(r.trades[0]["entry"], 130 * 1.001)   # filled at candle 11's open, not 100

    def test_entry_delay_stress_enters_one_candle_later(self):
        df = candles()
        r, _ = run(df, sig(40, {10: (95, 110)}), cfg(entry_delay_bars=1))
        self.assertEqual(r.trades[0]["entry_ts"], TS0 + 12 * TF)

    def test_funding_is_charged_for_each_settlement_while_held(self):
        df = candles(over={12: (100, 111, 99, 105)})       # target hit in candle 12 -> exit time = end of 12 = T13
        fund = (np.array([TS0 + 11 * TF, TS0 + 12 * TF, TS0 + 13 * TF], dtype=np.int64),
                np.array([0.5, 0.001, 0.002]))
        r, _ = run(df, sig(40, {10: (95, 110)}), cfg(max_funding_abs=1.0), fund)
        t = r.trades[0]
        # settlement exactly at entry is NOT paid; those at T12 and T13 (= exit) are.
        self.assertAlmostEqual(t["funding"], t["notional"] * (0.001 + 0.002), places=9)

    def test_negative_funding_is_received_unless_worst_case(self):
        df = candles(over={12: (100, 111, 99, 105)})
        fund = (np.array([TS0 + 4 * TF, TS0 + 12 * TF], dtype=np.int64), np.array([0.0, -0.001]))
        r1, _ = run(df, sig(40, {10: (95, 110)}), cfg(), fund)
        r2, _ = run(df, sig(40, {10: (95, 110)}), cfg(funding_worst_case=True), fund)
        self.assertLess(r1.trades[0]["funding"], 0)        # a credit
        self.assertGreater(r2.trades[0]["funding"], 0)     # stress mode charges it anyway
        self.assertGreater(r1.final_balance, r2.final_balance)

    def test_high_funding_blocks_entry(self):
        df = candles()
        fund = (np.array([TS0 + 5 * TF], dtype=np.int64), np.array([0.002]))
        r, _ = run(df, sig(40, {10: (95, 110)}), cfg(max_funding_abs=0.001), fund)
        self.assertEqual(len(r.trades), 0)
        self.assertEqual(r.skipped["funding"], 1)

    def test_liquidation_loses_the_margin(self):
        df = candles(over={12: (100, 100, 79, 80)})        # liq ~ 100.1*(1-0.2+0.005) = 80.6
        r, _ = run(df, sig(40, {10: (70, 130)}), self.c)   # stop is BELOW the liquidation price
        t = r.trades[0]
        self.assertEqual(t["outcome"], "LIQUIDATED")
        self.assertAlmostEqual(t["gross"], -t["notional"] / 5.0)      # whole isolated margin

    def test_more_slippage_never_helps(self):
        df = candles(over={12: (100, 111, 99, 105), 20: (100, 100, 90, 92)})
        s = sig(40, {10: (95, 110), 18: (95, 110)})
        finals = [run(df, s, cfg(slippage_pct=x))[0].final_balance for x in (0.0, 0.1, 0.5, 1.0)]
        self.assertEqual(finals, sorted(finals, reverse=True))

    def test_slot_limit_skips_extra_signals(self):
        df = candles()
        m = Market({"AAA": df, "BBB": df.copy()}, {}, TF)
        sigs = {"AAA": sig(40, {10: (95, 110)}), "BBB": sig(40, {10: (95, 110)})}
        r = simulate(m, sigs, cfg(max_simultaneous=1), int(df.ts_ms.iloc[1]), int(df.ts_ms.iloc[-1]) + TF)
        self.assertEqual(r.skipped["no_slot"], 1)
        self.assertEqual({t["symbol"] for t in r.trades}, {"AAA"})

    def test_window_end_closes_open_trades_with_costs(self):
        df = candles()
        r, _ = run(df, sig(40, {10: (95, 110)}), cfg(max_hold_hours=10_000))
        t = r.trades[0]
        self.assertEqual(t["outcome"], "END OF WINDOW")
        self.assertLess(t["net"], 0, "flat price: the round trip must cost fees + slippage")

    def test_entries_only_inside_the_window(self):
        df = candles()
        s = sig(40, {5: (95, 110), 30: (95, 110)})
        r, _ = run(df, s, self.c, start=TS0 + 20 * TF, end=TS0 + 40 * TF)
        self.assertTrue(all(t["entry_ts"] >= TS0 + 20 * TF for t in r.trades))
        self.assertEqual(len(r.trades), 1)

    def test_equity_curve_ends_at_final_balance(self):
        df = candles(over={12: (100, 111, 99, 105)})
        r, _ = run(df, sig(40, {10: (95, 110)}), self.c)
        self.assertAlmostEqual(r.eq[-1], r.final_balance)


class Baselines(unittest.TestCase):
    def test_buy_and_hold_flat_price_loses_only_costs(self):
        df = candles()
        m = Market({"AAA": df}, {}, TF)
        bh = metrics.buy_and_hold(m, cfg(), int(df.ts_ms.iloc[1]), int(df.ts_ms.iloc[-1]) + TF)
        expected = (0.999 * (1 - 0.001) / (1.001 * (1 + 0.001)) - 1) * 100    # slippage+fee in, slippage+fee out
        self.assertAlmostEqual(bh["summary"]["return_pct"], expected, places=6)

    def test_buy_and_hold_tracks_the_price(self):
        df = candles(price=100.0)
        df["close"] = np.linspace(100, 200, len(df))
        df["open"] = df["close"].shift(1).fillna(100)
        df["high"] = df["low"] = df["close"]
        m = Market({"AAA": df}, {}, TF)
        bh = metrics.buy_and_hold(m, cfg(taker_fee=0, slippage_pct=0), int(df.ts_ms.iloc[1]), int(df.ts_ms.iloc[-1]) + TF)
        self.assertAlmostEqual(bh["summary"]["return_pct"], (200 / df["open"].iloc[1] - 1) * 100, places=6)

    def test_random_baseline_is_reproducible(self):
        rng = np.random.default_rng(3)
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
        df = candles(600)
        df["open"] = np.r_[100, px[:-1]]; df["close"] = px
        df["high"] = np.maximum(df.open, df.close) * 1.004; df["low"] = np.minimum(df.open, df.close) * 0.996
        m = Market({"AAA": df}, {}, TF)
        strat = EmaRsi()
        sigs = m.signals(strat, {})
        c = cfg(risk_pct=1)
        a, b = int(df.ts_ms.iloc[70]), int(df.ts_ms.iloc[-1]) + TF
        sim = simulate(m, sigs, c, a, b)
        summ = metrics.summarize(sim, c, TF)
        summ["trades"] = max(summ["trades"], 5)
        r1 = metrics.random_baseline(m, sigs, c, a, b, summ, sim, n_sims=8, seed=5)
        r2 = metrics.random_baseline(m, sigs, c, a, b, summ, sim, n_sims=8, seed=5)
        self.assertEqual(r1["returns"], r2["returns"])


class StrategyIntegrity(unittest.TestCase):
    @staticmethod
    def walk(n=900, seed=11):
        rng = np.random.default_rng(seed)
        px = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
        df = candles(n)
        df["open"] = np.r_[100, px[:-1]]; df["close"] = px
        df["high"] = np.maximum(df.open, df.close) * (1 + rng.uniform(0, 0.006, n))
        df["low"] = np.minimum(df.open, df.close) * (1 - rng.uniform(0, 0.006, n))
        df["vol"] = rng.uniform(0.5, 2.0, n) * (1 + 1.5 * (rng.random(n) < 0.15))
        return df

    # Looser-than-default parameters, so each strategy actually fires on a random walk.
    LOOSE = {"ema_rsi": {"rsi_entry_max": 70, "vol_mult": 0.0},
             "trend_pullback": {"trend_len": 50, "rsi_buy": 50},
             "breakout": {"trend_len": 50, "breakout_len": 10, "exit_len": 5},
             "rules": {"trend_len": 50, "rsi_level": 55, "use_breakout": 1, "breakout_len": 8,
                       "use_exit_channel": 1, "exit_len": 5}}

    def each_strategy(self):
        self.assertEqual(set(self.LOOSE), set(STRATEGIES), "add LOOSE params here for every new strategy")
        for name, strat in STRATEGIES.items():
            yield name, strat

    def test_no_lookahead_for_every_strategy(self):
        """Signals computed with only the past must equal signals computed with the whole history."""
        df = self.walk(1600)
        for name, s in self.each_strategy():
            grid_corner = {k: v[-1] for k, v in s.param_grid.items()}
            for params in ({}, self.LOOSE[name], grid_corner):
                full = s.compute(df, params)
                for i in range(120, len(df), 37):
                    part = s.compute(df.iloc[: i + 1], params).iloc[-1]
                    row = full.iloc[i]
                    for col in ("enter", "exit"):
                        self.assertEqual(bool(part[col]), bool(row[col]), f"{name} {col} differs at {i} with {params}")
                    if row["enter"]:
                        for col in ("stop", "target"):
                            self.assertAlmostEqual(part[col], row[col], places=9, msg=f"{name} {col} at {i}")

    def test_every_strategy_fires_on_the_sample(self):
        df = self.walk(1600)
        for name, s in self.each_strategy():
            n = int(s.compute(df, self.LOOSE[name])["enter"].sum())
            self.assertGreater(n, 3, f"{name} produced only {n} signals; the sample can't test it")

    def test_live_signal_matches_backtest_signal_for_every_strategy(self):
        """The live bot calls signal(); the backtester uses compute(). They must agree."""
        df = self.walk(1000)
        for name, s in self.each_strategy():
            params = self.LOOSE[name]
            full = s.compute(df, params)
            hits = 0
            for i in range(s.warmup_bars(params), len(df)):
                live = s.signal("AAA", df.iloc[: i + 1], params)
                self.assertEqual(live is not None, bool(full["enter"].iloc[i]), f"{name} disagrees at {i}")
                hits += live is not None
            self.assertGreater(hits, 3, name)

    def test_no_signals_before_warmup(self):
        df = self.walk(1600)
        for name, s in self.each_strategy():
            for params in ({}, self.LOOSE[name]):
                c = s.compute(df, params)
                w = s.warmup_bars(params)
                self.assertFalse(c["enter"].iloc[: w - 1].any(), f"{name} signalled inside its warm-up")
                self.assertFalse(c["exit"].iloc[: w - 1].any(), f"{name} exited inside its warm-up")
                self.assertIsNone(s.signal("AAA", df.iloc[: w - 1], params))

    def test_grid_is_valid_and_within_reason(self):
        for name, s in self.each_strategy():
            if s.param_grid:
                from app.backtest.runs import expand_grid
                combos = expand_grid(s, s.default_params, s.param_grid)
                self.assertTrue(combos, name)
                self.assertLessEqual(len(combos), 700, f"{name}: a bigger grid means more chances to fit noise")

    def test_explain_funnel_is_monotonic(self):
        """Each funnel step can only keep or remove candidates, and the last step equals the entry signals."""
        df = self.walk(1600)
        for name, s in self.each_strategy():
            params = self.LOOSE[name]
            conds = s.explain(df, params)
            if conds is None:
                continue
            acc = np.ones(len(df), bool)
            counts = []
            for _, arr in conds:
                acc = acc & np.asarray(arr, bool)
                counts.append(int(acc.sum()))
            self.assertEqual(counts, sorted(counts, reverse=True), name)
            self.assertEqual(counts[-1], int(s.compute(df, params)["enter"].sum()), f"{name}: funnel end != signals")

    def test_entries_only_happen_inside_the_regime(self):
        df = self.walk(1600)
        for name, s in self.each_strategy():
            c = s.compute(df, self.LOOSE[name])
            ctx = s.context(df, self.LOOSE[name])
            self.assertFalse((c["enter"].to_numpy() & ~ctx).any(), f"{name} enters outside its own regime filter")

    def test_same_regime_random_baseline(self):
        df = self.walk(1600)
        m = Market({"AAA": df}, {}, TF)
        strat = STRATEGIES["breakout"]
        params = self.LOOSE["breakout"]
        sigs = m.signals(strat, params)
        c = cfg(risk_pct=1)
        a, b = int(df.ts_ms.iloc[300]), int(df.ts_ms.iloc[-1]) + TF
        sim = simulate(m, sigs, c, a, b)
        summ = metrics.summarize(sim, c, TF)
        self.assertGreater(summ["trades"], 5)
        r1 = metrics.random_baseline(m, sigs, c, a, b, summ, sim, n_sims=20, seed=3, regime=True)
        r2 = metrics.random_baseline(m, sigs, c, a, b, summ, sim, n_sims=20, seed=3, regime=True)
        self.assertEqual(r1["returns"], r2["returns"], "must be reproducible")
        self.assertTrue(0.4 * summ["trades"] < r1["mean_trades"] < 2.5 * summ["trades"],
                        f"random runs should trade about as often as the strategy ({r1['mean_trades']} vs {summ['trades']})")
        # entries drawn only inside the regime can never exceed the number of in-regime candles
        self.assertLess(r1["prob_per_bar"], 0.5 + 1e-9)

    def test_strategy_with_only_signal_still_works(self):
        """A strategy that only implements signal()/exit_signal() gets the generic (slow) compute()."""
        class Momentum(Strategy):
            name = "mom"
            def signal(self, symbol, df, params):
                c = df["close"].to_numpy()
                if len(c) >= 3 and c[-1] > c[-2] * 1.01:
                    return Signal(symbol, float(c[-1]), float(c[-1] * 0.95), float(c[-1] * 1.10), 1.0, 50.0,
                                  df["ts"].iloc[-1].isoformat())
                return None
            def exit_signal(self, df, params):
                c = df["close"].to_numpy()
                return bool(len(c) >= 2 and c[-1] < c[-2] * 0.99)

        df = self.walk(300)
        out = Momentum().compute(df, {})
        c = df["close"].to_numpy()
        want_enter = np.zeros(len(c), bool)
        want_exit = np.zeros(len(c), bool)
        want_enter[2:] = c[2:] > c[1:-1] * 1.01
        want_exit[1:] = c[1:] < c[:-1] * 0.99
        self.assertTrue((out["enter"].to_numpy() == want_enter).all())
        self.assertTrue((out["exit"].to_numpy() == want_exit).all())
        self.assertGreater(int(want_enter.sum()), 3)


if __name__ == "__main__":
    unittest.main()
