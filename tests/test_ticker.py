"""The price strip's parsing of Bitget's tickers response (no network)."""
import unittest

from app.exchange.bitget import parse_tickers

ROWS = [{"symbol": "SOLUSDT", "lastPr": "118.379", "change24h": "0.0125"},
        {"symbol": "LINKUSDT", "lastPr": "15.4", "change24h": "-0.05"},
        {"symbol": "AVAXUSDT", "lastPr": "11.5", "change24h": ""},
        {"symbol": "DEADUSDT", "lastPr": ""}]


class ParseTickers(unittest.TestCase):
    def test_keeps_the_requested_order_and_converts_the_change_to_percent(self):
        out = parse_tickers(ROWS, ["LINKUSDT", "SOLUSDT"])
        self.assertEqual([r["symbol"] for r in out], ["LINKUSDT", "SOLUSDT"])
        self.assertAlmostEqual(out[0]["change_pct"], -5.0)
        self.assertAlmostEqual(out[1]["change_pct"], 1.25)
        self.assertEqual(out[1]["price"], 118.379)

    def test_missing_change_is_none_and_unknown_or_priceless_symbols_are_skipped(self):
        out = parse_tickers(ROWS, ["AVAXUSDT", "NOPEUSDT", "DEADUSDT"])
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0]["change_pct"])


if __name__ == "__main__":
    unittest.main()
