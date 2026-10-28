"""Tests for saved strategies, per-bot accounts and the one-time migration of the original
single bot. Uses a throwaway SQLite file; nothing touches the network."""
import json
import os
import tempfile
import unittest
from unittest import mock

from app import db as database
from app import strategy_store as store
from app.futures import engine
from app.strategies import STRATEGIES


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(database, "DB_PATH", os.path.join(self.tmp.name, "t.db"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        database.init()

    def add_trade(self, bot_id, net, status="closed"):
        with database.db() as c:
            c.execute("INSERT INTO futures_trades(symbol, strategy, status, entry_time, entry_price, stop, target,"
                      " units, notional, margin, fee_open, exit_time, net_pnl, balance_after, bot_id) "
                      "VALUES('SOLUSDT','x',?,'2026-01-01T00:00:00+00:00',10,9,12,1,10,2,0.01,"
                      "'2026-01-02T00:00:00+00:00',?,?,?)", (status, net, 100 + net, bot_id))

    def new_strategy(self, name="s", stype="ema_rsi"):
        return store.save_strategy({"name": name, "type": stype, "params": {}, "pairs": ["SOLUSDT"],
                                    "timeframe": "4h", "risk": {"starting_capital": 200.0}})


class Migration(StoreTestCase):
    def legacy(self):
        with database.db() as c:
            for k, v in {"futures.strategy": "breakout", "futures.enabled": True, "futures.pairs": ["SOLUSDT"],
                         "futures.max_hold_hours": 240, "strategy.breakout": {"trend_len": 150}}.items():
                c.execute("INSERT INTO settings(key, value) VALUES(?,?)", (k, json.dumps(v)))
            c.execute("INSERT INTO futures_state(key, value) VALUES('balance','95.5')")
            c.execute("INSERT INTO events(ts, source, level, message) VALUES('t','futures','info','old')")
        self.add_trade(None, -1.0)

    def test_legacy_bot_becomes_strategy_and_bot_with_state_and_trades(self):
        self.legacy()
        store.migrate_legacy()
        (s,), (b,) = store.list_strategies(), store.list_bots()
        self.assertEqual(s["type"], "breakout")
        self.assertEqual(s["params"]["trend_len"], 150)                  # tuned value kept
        self.assertEqual(s["params"]["breakout_len"], STRATEGIES["breakout"].default_params["breakout_len"])  # rest from defaults
        self.assertEqual(s["risk"]["max_hold_hours"], 240)
        self.assertEqual(b["status"], "running")
        self.assertEqual(b["cfg"]["params"]["trend_len"], 150)
        self.assertEqual(engine.get_state(b["id"], "balance"), "95.5")
        with database.db() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) n FROM futures_trades WHERE bot_id=?", (b["id"],)).fetchone()["n"], 1)
            self.assertEqual(c.execute("SELECT COUNT(*) n FROM events WHERE bot_id IS NULL").fetchone()["n"], 0)

    def test_runs_once(self):
        self.legacy()
        store.migrate_legacy()
        store.migrate_legacy()
        self.assertEqual(len(store.list_strategies()), 1)
        self.assertEqual(len(store.list_bots()), 1)

    def test_fresh_install_migrates_nothing(self):
        store.migrate_legacy()
        self.assertEqual(store.list_strategies(), [])

    def test_disabled_legacy_bot_stays_paused(self):
        self.legacy()
        with database.db() as c:
            c.execute("UPDATE settings SET value='false' WHERE key='futures.enabled'")
        store.migrate_legacy()
        self.assertEqual(store.list_bots()[0]["status"], "paused")


class Bots(StoreTestCase):
    def test_bot_snapshot_is_unaffected_by_later_strategy_edits(self):
        sid = self.new_strategy()
        bid = store.create_bot(sid)
        s = store.get_strategy(sid)
        store.save_strategy({**s, "pairs": ["AVAXUSDT"], "risk": {**s["risk"], "leverage": 2}}, sid)
        cfg = store.get_bot(bid)["cfg"]
        self.assertEqual(cfg["pairs"], ["SOLUSDT"])
        self.assertEqual(cfg["leverage"], 5)

    def test_each_bot_has_its_own_account(self):
        a, b = store.create_bot(self.new_strategy("a")), store.create_bot(self.new_strategy("b", "breakout"))
        self.add_trade(a, 5.0)
        self.assertEqual(engine.summary(a, store.get_bot(a)["cfg"])["trades"], 1)
        self.assertEqual(engine.summary(b, store.get_bot(b)["cfg"])["trades"], 0)
        engine.set_state(a, "balance", 50)
        self.assertEqual(float(engine.get_state(b, "balance")), 200.0)   # starting capital, untouched

    def test_reset_only_wipes_that_bot(self):
        a, b = store.create_bot(self.new_strategy("a")), store.create_bot(self.new_strategy("b"))
        self.add_trade(a, 5.0)
        self.add_trade(b, 7.0)
        engine.set_state(a, "balance", 999)
        engine.reset_account(a)
        self.assertEqual(engine.summary(a, store.get_bot(a)["cfg"])["trades"], 0)
        self.assertEqual(float(engine.get_state(a, "balance")), 200.0)
        self.assertEqual(engine.summary(b, store.get_bot(b)["cfg"])["trades"], 1)

    def test_paused_bot_is_not_scanned(self):
        bid = store.create_bot(self.new_strategy())
        store.set_bot_status(bid, "paused")
        self.assertEqual(engine.scan_once(bid), "paused")
        self.assertEqual(engine.scan_once(9999), "missing")

    def test_scan_is_due_only_after_its_own_interval(self):
        bid = store.create_bot(self.new_strategy())
        bot = store.get_bot(bid)
        self.assertTrue(engine.scan_due(bot))                             # never scanned
        engine.set_state(bid, "last_scan", engine.iso(engine.utcnow()))
        self.assertFalse(engine.scan_due(bot))
        engine.set_state(bid, "last_scan", engine.iso(engine.utcnow() - engine.timedelta(minutes=16)))
        self.assertTrue(engine.scan_due(bot))                             # scan_minutes default is 15

    def test_all_pairs_covers_every_saved_strategy(self):
        store.save_strategy({"name": "x", "type": "ema_rsi", "params": {}, "pairs": ["DOGEUSDT"],
                             "timeframe": "4h", "risk": {}})
        self.assertIn("DOGEUSDT", store.all_pairs())
        self.assertIn("SOLUSDT", store.all_pairs())


if __name__ == "__main__":
    unittest.main()
