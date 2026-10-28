"""Tests for the small Jinja filters defined directly in app/main.py.

Importing app.main is safe in a test: creating the FastAPI app and registering routes/filters
has no side effects on its own — the background runners (the futures engine)
only start from the ASGI lifespan hook, which nothing here triggers.
"""
import unittest

from app.main import iso_local, pair, pair_list, tv_url


class IsoLocal(unittest.TestCase):
    """Every stored timestamp is UTC; every displayed one should be UK local time — GMT in
    winter, BST in summer, switching on its own rather than a fixed offset."""

    def test_summer_utc_timestamp_shown_as_bst(self):
        self.assertEqual(iso_local("2026-07-01T12:00:00+00:00"), "2026-07-01 13:00")

    def test_winter_utc_timestamp_shown_as_gmt(self):
        self.assertEqual(iso_local("2026-01-01T12:00:00+00:00"), "2026-01-01 12:00")

    def test_naive_string_with_no_offset_is_treated_as_utc(self):
        # every ISO string this app stores has an offset (datetime.now(timezone.utc).isoformat())
        # but this covers the field defensively rather than crashing on an unexpected one
        self.assertEqual(iso_local("2026-07-01T12:00:00"), "2026-07-01 13:00")

    def test_blank_or_missing_is_a_dash_not_a_crash(self):
        self.assertEqual(iso_local(""), "–")
        self.assertEqual(iso_local(None), "–")


class PairLinks(unittest.TestCase):
    """Every pair links to its Bitget perpetual on TradingView."""

    def test_url_targets_the_bitget_perpetual_and_asks_for_the_dark_chart(self):
        self.assertEqual(tv_url("SOLUSDT"), "https://www.tradingview.com/chart/?symbol=BITGET%3ASOLUSDT.P&theme=dark")

    def test_link_shows_the_short_name_by_default_and_opens_a_new_tab_safely(self):
        html = str(pair("SOLUSDT"))
        self.assertIn(">SOL</a>", html)
        self.assertIn('target="_blank"', html)
        self.assertIn("noopener", html)
        self.assertIn(">SOLUSDT</a>", str(pair("SOLUSDT", short=False)))

    def test_list_is_comma_separated_links(self):
        html = str(pair_list(["SOLUSDT", "AVAXUSDT"]))
        self.assertEqual(html.count("<a "), 2)
        self.assertIn("</a>, <a ", html)

    def test_symbols_are_escaped(self):
        self.assertNotIn("<script>", str(pair("<script>USDT")))
