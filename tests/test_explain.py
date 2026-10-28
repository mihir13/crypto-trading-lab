"""Plain-English strategy summaries (app/explain.py and each strategy's plain())."""
import unittest

from app import explain
from app import strategy_store as store
from app.exchange.bitget import parse_symbol_list
from app.strategies import STRATEGIES


def strat(stype, params=None, **risk):
    s = STRATEGIES[stype]
    return {"id": 1, "name": "t", "type": stype, "params": {**s.default_params, **(params or {})}, "pairs": ["SOLUSDT", "AVAXUSDT"],
            "timeframe": "4h", "risk": {**store.RISK_DEFAULTS, **risk}, "notes": ""}


def row(**kw):
    return {"audit": None, "oos": None, "live": None, "stale": False, "next": "Run an audit", "tone": "warn", **kw}


class Plain(unittest.TestCase):
    def test_every_strategy_type_explains_itself_completely(self):
        for name, s in STRATEGIES.items():
            p = s.plain(s.default_params)
            for key in ("tagline", "idea"):
                self.assertTrue(p[key], f"{name} has no {key}")
            for key in ("buy", "sell", "watch"):
                self.assertTrue(p[key], f"{name} has no {key} list")
            for t in p["terms"]:
                self.assertIn(t, explain.GLOSSARY, f"{name} mentions an undefined term {t}")

    def test_the_text_uses_the_real_parameter_values(self):
        p = STRATEGIES["breakout"].plain({"trend_len": 150, "breakout_len": 33, "exit_len": 7})
        text = " ".join(p["buy"] + p["sell"])
        for n in ("150", "33", "7"):
            self.assertIn(n, text)

    def test_breakout_with_a_huge_target_says_there_is_no_target(self):
        self.assertIn("no profit target", " ".join(STRATEGIES["breakout"].plain({"risk_reward": 10})["sell"]))
        self.assertNotIn("no profit target", " ".join(STRATEGIES["breakout"].plain({"risk_reward": 3})["sell"]))

    def test_custom_rules_describe_only_what_is_switched_on(self):
        r = STRATEGIES["rules"]
        off = {"use_trend": 0, "use_rsi": 0, "use_breakout": 0, "use_cross": 0}
        p = r.plain({**off, "use_breakout": 1, "breakout_len": 25, "use_exit_rsi": 1, "exit_rsi_level": 80})
        self.assertEqual(len(p["buy"]), 1)
        self.assertIn("25", p["buy"][0])
        self.assertIn("80", " ".join(p["sell"]))
        self.assertIn("breakout", p["tagline"])
        self.assertIn("never buy", r.plain(off)["buy"][0])

    def test_rules_glossary_terms_are_all_defined(self):
        p = STRATEGIES["rules"].plain({"use_trend": 1, "use_rsi": 1, "use_breakout": 1, "use_cross": 1, "use_exit_rsi": 1})
        self.assertTrue(set(p["terms"]) <= set(explain.GLOSSARY))


class Sections(unittest.TestCase):
    def test_risk_lines_use_the_account_settings(self):
        text = " ".join(explain.risk_lines(strat("breakout", starting_capital=200.0, risk_pct=2.0, leverage=3)))
        for x in ("SOL, AVAX", "4h", "2%", "$4.00", "$200", "3x"):
            self.assertIn(x, text)

    def test_track_record_without_tests_says_so(self):
        self.assertIn("Not backtested", explain.track_lines(row(), None)[0])
        self.assertIn("settings have changed", explain.track_lines(row(stale=True), None)[0])

    def test_track_record_reports_backtest_out_of_sample_and_live(self):
        audit = {"years": 4.7, "trades": 340, "return_pct": 238.0, "bh_return_pct": -50.0, "level": "moderate"}
        lines = explain.track_lines(row(audit=audit, oos={"return_pct": 40.0, "trades": 80, "level": "none"},
                                        live={"days": 12, "trades": 5, "return_pct": -3.0}),
                                    {"summary": {"max_dd_pct": -33.0, "win_rate": 40.0}})
        text = " ".join(lines)
        for x in ("4.7 years", "340 trades", "+238%", "-50%", "33%", "40% of its trades", "some evidence", "+40%", "no reliable edge",
                  "12 days", "too few trades"):
            self.assertIn(x, text)

    def test_summary_has_a_glossary_with_the_always_terms(self):
        s = explain.summary(strat("breakout"), row())
        terms = [t for t, _ in s["glossary"]]
        self.assertIn("Leverage", terms)
        self.assertTrue(all(defn for _, defn in s["glossary"]))


class SymbolList(unittest.TestCase):
    def test_lists_usdt_perpetuals_busiest_first_and_skips_junk(self):
        rows = [{"symbol": "AAAUSDT", "lastPr": "1", "usdtVolume": "10", "change24h": "0.1"},
                {"symbol": "BBBUSDT", "lastPr": "2", "usdtVolume": "500", "change24h": ""},
                {"symbol": "CCCUSDC", "lastPr": "3", "usdtVolume": "999"},
                {"symbol": "USDT", "lastPr": "1", "usdtVolume": "999"},
                {"symbol": "DDDUSDT", "lastPr": "", "usdtVolume": "5"}]
        out = parse_symbol_list(rows)
        self.assertEqual([r["symbol"] for r in out], ["BBBUSDT", "AAAUSDT"])
        self.assertEqual(out[0]["base"], "BBB")
        self.assertIsNone(out[0]["change_pct"])
        self.assertAlmostEqual(out[1]["change_pct"], 10.0)


if __name__ == "__main__":
    unittest.main()
