"""Tests for turning a submitted strategy form into validated fields (app.main._parse_strategy_form)."""
import unittest
from unittest import mock

from app.main import _parse_strategy_form


def form(**kw):
    base = {"name": "t", "type": "rules", "pairs": "solusdt, AVAXUSDT ,solusdt", "timeframe": "4h"}
    return {**base, **kw}


MARKET = [{"symbol": x, "base": x[:-4], "volume": 1.0, "price": 1.0, "change_pct": 0.0}
          for x in ("SOLUSDT", "AVAXUSDT", "ETHUSDT", "DOGEUSDT")]


class ParseStrategyForm(unittest.TestCase):
    def setUp(self):
        p = mock.patch("app.main.market_symbols", return_value=[])       # no network in tests
        p.start()
        self.addCleanup(p.stop)
    def test_checkboxes_present_are_on_and_absent_are_off(self):
        fields, err = _parse_strategy_form(form(p_use_trend="on", p_use_breakout="on"))
        self.assertEqual(err, "")
        self.assertEqual(fields["params"]["use_trend"], 1)
        self.assertEqual(fields["params"]["use_breakout"], 1)
        self.assertEqual(fields["params"]["use_rsi"], 0)         # the default is ON, but unticked means off

    def test_no_entry_condition_is_rejected_with_a_reason(self):
        fields, err = _parse_strategy_form(form())
        self.assertIsNone(fields)
        self.assertIn("entry condition", err)

    def test_pairs_are_cleaned_uppercased_and_deduplicated(self):
        fields, _ = _parse_strategy_form(form(p_use_trend="on", pairs="solusdt, AVAXUSDT ,solusdt, nonsense"))
        self.assertEqual(fields["pairs"], ["SOLUSDT", "AVAXUSDT"])

    def test_no_valid_pair_is_an_error(self):
        fields, err = _parse_strategy_form(form(p_use_trend="on", pairs="hello"))
        self.assertIsNone(fields)
        self.assertIn("pair", err)

    def test_numbers_are_clamped_and_bad_input_falls_back_to_the_default(self):
        fields, _ = _parse_strategy_form(form(p_use_trend="on", leverage="500", risk_pct="abc", p_trend_len="150"))
        self.assertEqual(fields["risk"]["leverage"], 20)          # capped
        self.assertEqual(fields["risk"]["risk_pct"], 3.0)         # default
        self.assertEqual(fields["params"]["trend_len"], 150)

    def test_name_and_type_are_required(self):
        self.assertIn("name", _parse_strategy_form(form(name=" "))[1])
        self.assertIn("type", _parse_strategy_form(form(type="nope"))[1])

    def test_hand_written_strategies_still_parse_plain_numbers(self):
        fields, err = _parse_strategy_form(form(type="breakout", p_trend_len="120"))
        self.assertEqual(err, "")
        self.assertEqual(fields["params"]["trend_len"], 120)


class PairChecks(unittest.TestCase):
    def check(self, pairs, market=MARKET):
        with mock.patch("app.main.market_symbols", return_value=market):
            return _parse_strategy_form(form(p_use_trend="on", pairs=pairs))

    def test_a_pair_bitget_does_not_list_is_rejected_by_name(self):
        fields, err = self.check("SOLUSDT, FAKEUSDT")
        self.assertIsNone(fields)
        self.assertIn("FAKEUSDT", err)

    def test_listed_pairs_pass(self):
        fields, err = self.check("ethusdt, dogeusdt")
        self.assertEqual((err, fields["pairs"]), ("", ["ETHUSDT", "DOGEUSDT"]))

    def test_when_the_exchange_cannot_be_reached_the_format_check_alone_applies(self):
        fields, err = self.check("FAKEUSDT", market=[])
        self.assertEqual(fields["pairs"], ["FAKEUSDT"])

    def test_too_many_pairs_are_rejected(self):
        many = ", ".join(f"C{i}USDT" for i in range(13))
        fields, err = self.check(many, market=[])
        self.assertIsNone(fields)
        self.assertIn("at most", err)


if __name__ == "__main__":
    unittest.main()
