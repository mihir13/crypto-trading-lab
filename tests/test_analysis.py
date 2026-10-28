"""Tests for app/analysis.py: which results count for a strategy's current settings, the evidence
pipeline stages, and matching live trades to backtest trades."""
import json
import os
import tempfile
import unittest
from unittest import mock

from app import analysis as an
from app import db as database
from app import strategy_store as store


def strategy(**over):
    base = {"id": 1, "name": "s", "type": "breakout", "params": {"trend_len": 100}, "pairs": ["SOLUSDT", "AVAXUSDT"],
            "timeframe": "4h", "risk": dict(store.RISK_DEFAULTS), "notes": ""}
    return {**base, **over}


def full(s):
    """The strategy as store.get_strategy() returns it (params merged with the type's defaults)."""
    from app.strategies import STRATEGIES
    return {**s, "params": {**STRATEGIES[s["type"]].default_params, **s["params"]}}


class SettingsMatching(unittest.TestCase):
    def test_a_run_of_the_same_settings_matches(self):
        s = full(strategy())
        self.assertTrue(an.run_matches(an.backtest_config(s), s))

    def test_pair_order_and_unset_defaults_do_not_matter(self):
        s = full(strategy())
        rc = an.backtest_config(s)
        rc["symbols"] = list(reversed(rc["symbols"]))
        rc["params"] = {"trend_len": 100}                     # defaults left out
        self.assertTrue(an.run_matches(rc, s))

    def test_any_change_to_settings_breaks_the_match(self):
        s = full(strategy())
        for change in ({"params": {**s["params"], "breakout_len": 41}}, {"pairs": ["SOLUSDT"]},
                       {"timeframe": "1h"}, {"risk": {**s["risk"], "slippage_pct": 0.5}},
                       {"risk": {**s["risk"], "leverage": 10}}):
            self.assertFalse(an.run_matches(an.backtest_config(s), full({**s, **change})), change)

    def test_float_noise_from_a_form_round_trip_still_matches(self):
        s = full(strategy())
        rc = an.backtest_config(s)
        rc["cfg"]["taker_fee"] = 0.06 / 100                  # 0.0006 via a percent field
        self.assertTrue(an.run_matches(rc, s))

    def test_bot_snapshot_matches_until_the_strategy_changes_but_scan_interval_is_ignored(self):
        s = full(strategy())
        cfg = store.strategy_config(s)
        self.assertTrue(an.bot_matches({**cfg, "scan_minutes": 99}, s))
        self.assertFalse(an.bot_matches(cfg, full({**s, "params": {**s["params"], "exit_len": 5}})))
        self.assertFalse(an.bot_matches(cfg, full({**s, "risk": {**s["risk"], "risk_pct": 1.0}})))


def brief(level="moderate", ret=10.0, trades=100):
    return {"level": level, "return_pct": ret, "trades": trades}


def live(trades=40, ret=5.0):
    return {"trades": trades, "return_pct": ret}


class Stages(unittest.TestCase):
    def test_each_gate_needs_the_one_before(self):
        self.assertEqual(an.stage_of(None, None, None), 0)
        self.assertEqual(an.stage_of(brief("none"), brief(), live()), 1)          # fails the audit: later results ignored
        self.assertEqual(an.stage_of(brief(ret=-3), brief(), live()), 1)          # loses money
        self.assertEqual(an.stage_of(brief(), None, live()), 2)
        self.assertEqual(an.stage_of(brief(), brief("none"), live()), 2)
        self.assertEqual(an.stage_of(brief(), brief(), None), 3)
        self.assertEqual(an.stage_of(brief(), brief(), live(trades=an.MIN_LIVE_TRADES - 1)), 3)
        self.assertEqual(an.stage_of(brief(), brief(), live(ret=-1)), 3)
        self.assertEqual(an.stage_of(brief(), brief(), live()), 4)
        self.assertEqual(an.stage_of(brief("weak"), brief("weak"), live()), 4)   # weak evidence still passes

    def test_next_step_tells_you_what_to_do(self):
        a, o = brief(), brief()
        self.assertIn("audit", an.next_step(0, None, None, None, False, False)[0])
        self.assertIn("Settings changed", an.next_step(0, None, None, None, True, False)[0])
        self.assertEqual(an.next_step(1, brief("none"), None, None, False, False)[1], "bad")
        self.assertIn("out-of-sample", an.next_step(2, a, None, None, False, False)[0])
        self.assertIn("Start a paper bot", an.next_step(3, a, o, None, False, False)[0])
        self.assertIn("5/30", an.next_step(3, a, o, live(trades=5), False, True)[0])
        self.assertEqual(an.next_step(3, a, o, live(trades=50, ret=-2), False, True)[1], "bad")
        self.assertEqual(an.next_step(4, a, o, live(), False, True)[1], "good")


class MatchTrades(unittest.TestCase):
    TF = 4 * 3600 * 1000

    def test_live_trade_pairs_with_the_backtest_trade_from_the_same_signal(self):
        # signal candle opened at t0; the backtest enters at the NEXT candle's open, t0 + tf
        live_t = [{"symbol": "SOL", "signal_ms": 1000}, {"symbol": "SOL", "signal_ms": 5 * self.TF},
                  {"symbol": "AVAX", "signal_ms": None}]
        bt = [{"symbol": "SOL", "entry_ts": 1000 + self.TF}, {"symbol": "SOL", "entry_ts": 9 * self.TF},
              {"symbol": "AVAX", "entry_ts": 1000 + self.TF}]
        m = an.match_trades(live_t, bt, self.TF)
        self.assertEqual(len(m["matched"]), 1)
        self.assertEqual(m["matched"][0][0]["signal_ms"], 1000)
        self.assertEqual(len(m["live_only"]), 2)                                   # different signal, and one with no signal time
        self.assertEqual({(t["symbol"], t["entry_ts"]) for t in m["bt_only"]},
                         {("SOL", 9 * self.TF), ("AVAX", 1000 + self.TF)})

    def test_the_same_symbol_on_different_pairs_never_cross_matches(self):
        m = an.match_trades([{"symbol": "SOL", "signal_ms": 0}], [{"symbol": "AVAX", "entry_ts": self.TF}], self.TF)
        self.assertEqual(m["matched"], [])

    def test_one_backtest_trade_is_only_used_once(self):
        live_t = [{"symbol": "SOL", "signal_ms": 0}, {"symbol": "SOL", "signal_ms": 0}]
        m = an.match_trades(live_t, [{"symbol": "SOL", "entry_ts": self.TF}], self.TF)
        self.assertEqual((len(m["matched"]), len(m["live_only"])), (1, 1))


class ReplayWindow(unittest.TestCase):
    def test_window_starts_at_the_earliest_of_bot_start_and_live_signals(self):
        self.assertEqual(an.window_start_ms(1000, []), 1000)
        self.assertEqual(an.window_start_ms(1000, [{"signal_ms": 400}, {"signal_ms": 900}, {"signal_ms": None}]), 400)
        self.assertEqual(an.window_start_ms(1000, [{"signal_ms": 5000}]), 1000)


class StaleRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        p = mock.patch.object(database, "DB_PATH", os.path.join(self.tmp.name, "t.db"))
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        database.init()

    def add_run(self, sid, config, kind="audit", level="moderate", ret=10.0):
        with database.db() as c:
            c.execute("INSERT INTO bt_runs(created, kind, status, config, brief, strategy_id) VALUES('t',?,?,?,?,?)",
                      (kind, "done", json.dumps(config), json.dumps(brief(level, ret)), sid))

    def make(self):
        sid = store.save_strategy({"name": "s", "type": "breakout", "params": {}, "pairs": ["SOLUSDT"],
                                   "timeframe": "4h", "risk": {}})
        return store.get_strategy(sid)

    def test_only_runs_of_current_settings_count_and_older_ones_are_flagged(self):
        s = self.make()
        old = an.backtest_config(s)
        self.add_run(s["id"], old)
        s2 = {**s, "params": {**s["params"], "breakout_len": 55}}
        store.save_strategy(s2, s["id"])
        row = an.strategy_row(store.get_strategy(s["id"]))
        self.assertIsNone(row["audit"])
        self.assertTrue(row["stale"])
        self.assertEqual(row["stage"], 0)
        self.add_run(s["id"], an.backtest_config(store.get_strategy(s["id"])))     # now a current audit
        row = an.strategy_row(store.get_strategy(s["id"]))
        self.assertEqual(row["stage"], 2)                                          # audit passes; no OOS yet
        self.assertIn("out-of-sample", row["next"])

    def test_a_bot_with_older_settings_does_not_count_as_live_evidence(self):
        s = self.make()
        self.add_run(s["id"], an.backtest_config(s))
        self.add_run(s["id"], an.backtest_config(s), kind="oos")
        store.create_bot(s["id"])
        self.assertIsNotNone(an.strategy_row(store.get_strategy(s["id"]))["live"])
        store.save_strategy({**s, "pairs": ["SOLUSDT", "AVAXUSDT"]}, s["id"])
        row = an.strategy_row(store.get_strategy(s["id"]))
        self.assertIsNone(row["live"])
        self.assertEqual(row["other_bots"], 1)


if __name__ == "__main__":
    unittest.main()
