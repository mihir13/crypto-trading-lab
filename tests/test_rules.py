"""Tests for the switchable-rule strategy type (app/strategies/rules.py). The shared integrity
checks (no lookahead, live == backtest, warm-up, funnel) live in test_backtest.StrategyIntegrity."""
import unittest

from app.strategies import STRATEGIES
from tests import test_backtest

R = STRATEGIES["rules"]
OFF = {k: 0 for k in R.ENTRY_FLAGS}


def entries(df, **params):
    return int(R.compute(df, {**OFF, **params})["enter"].sum())


class Rules(unittest.TestCase):
    df = test_backtest.StrategyIntegrity.walk(1600)

    def test_no_entry_condition_means_no_trades(self):
        self.assertEqual(entries(self.df), 0)
        self.assertIn("at least one entry", R.check(OFF))

    def test_each_extra_condition_can_only_remove_entries(self):
        a = entries(self.df, use_trend=1, trend_len=50)
        b = entries(self.df, use_trend=1, trend_len=50, use_rsi=1, rsi_level=60)
        c = entries(self.df, use_trend=1, trend_len=50, use_rsi=1, rsi_level=60, use_breakout=1, breakout_len=10)
        self.assertGreater(a, 0)
        self.assertLessEqual(b, a)
        self.assertLessEqual(c, b)

    def test_rsi_direction_switch_flips_the_condition(self):
        below = R.compute(self.df, {**OFF, "use_rsi": 1, "rsi_above": 0, "rsi_level": 50})["enter"]
        above = R.compute(self.df, {**OFF, "use_rsi": 1, "rsi_above": 1, "rsi_level": 50})["enter"]
        self.assertGreater(below.sum(), 0)
        self.assertGreater(above.sum(), 0)
        self.assertEqual(int((below & above).sum()), 0)                  # can't be both below and above 50

    def test_fresh_cross_is_a_subset_of_the_cross_state(self):
        state = R.compute(self.df, {**OFF, "use_cross": 1, "ema_fast": 5, "ema_slow": 20})["enter"]
        fresh = R.compute(self.df, {**OFF, "use_cross": 1, "ema_fast": 5, "ema_slow": 20, "cross_fresh": 1})["enter"]
        self.assertTrue((fresh & ~state).sum() == 0)
        self.assertLess(fresh.sum(), state.sum())

    def test_exit_rules_only_fire_when_switched_on(self):
        base = {**OFF, "use_trend": 1, "trend_len": 50}
        self.assertEqual(int(R.compute(self.df, base)["exit"].sum()), 0)
        self.assertGreater(int(R.compute(self.df, {**base, "use_exit_channel": 1, "exit_len": 5})["exit"].sum()), 0)
        self.assertGreater(int(R.compute(self.df, {**base, "use_exit_rsi": 1, "exit_rsi_level": 60})["exit"].sum()), 0)

    def test_check_rejects_impossible_settings(self):
        self.assertIn("fast EMA", R.check({**OFF, "use_cross": 1, "ema_fast": 30, "ema_slow": 10}))
        self.assertIn("greater than zero", R.check({"atr_stop_mult": 0}))
        self.assertIsNone(R.check({}))                                   # the defaults are usable

    def test_grid_tunes_only_switched_on_rules(self):
        g = R.grid_for({**OFF, "use_breakout": 1, "breakout_len": 40})
        self.assertEqual(set(g), {"breakout_len", "atr_stop_mult"})
        self.assertEqual(g["breakout_len"], [28, 40, 56])
        many = R.grid_for({k: 1 for k in R.ENTRY_FLAGS})
        self.assertLessEqual(len(many), 4)                               # three rules at most + the stop

    def test_describe_mentions_what_is_on(self):
        text = R.describe({**OFF, "use_trend": 1, "trend_len": 200, "use_exit_rsi": 1, "exit_rsi_level": 75})
        self.assertIn("EMA 200", text)
        self.assertIn("RSI rises above 75", text)
        self.assertIn("No entry condition", R.describe(OFF))


if __name__ == "__main__":
    unittest.main()
