"""Tests for app/pagination.py — pure functions, no DB. Shared by every list page (a live
bot's closed trades, and any list added later), so a bug here would misbehave the same way everywhere at once.
"""
import unittest

from app.pagination import DEFAULT_PAGE_SIZE, PAGE_SIZES, page_meta, pager_links, paginate


class PageMeta(unittest.TestCase):
    def test_first_page_of_an_exact_multiple(self):
        m = page_meta(total=50, page=1, per_page=25)
        self.assertEqual((m["page"], m["pages"], m["offset"], m["start"], m["end"]), (1, 2, 0, 1, 25))

    def test_last_page_partial(self):
        m = page_meta(total=53, page=3, per_page=25)
        self.assertEqual((m["page"], m["pages"], m["offset"], m["start"], m["end"]), (3, 3, 50, 51, 53))

    def test_page_beyond_the_last_one_clamps_down_to_it(self):
        m = page_meta(total=53, page=999, per_page=25)
        self.assertEqual(m["page"], 3)

    def test_page_below_one_clamps_up_to_one(self):
        self.assertEqual(page_meta(total=53, page=0, per_page=25)["page"], 1)
        self.assertEqual(page_meta(total=53, page=-5, per_page=25)["page"], 1)

    def test_zero_total_is_still_one_page_with_nothing_to_show(self):
        m = page_meta(total=0, page=1, per_page=25)
        self.assertEqual((m["pages"], m["start"], m["end"]), (1, 0, 0))

    def test_invalid_per_page_falls_back_to_the_default(self):
        # e.g. someone hand-edits the URL to per_page=13, which isn't one of the four offered
        self.assertEqual(page_meta(total=100, page=1, per_page=13)["per_page"], DEFAULT_PAGE_SIZE)

    def test_every_advertised_page_size_is_accepted_unchanged(self):
        for n in PAGE_SIZES:
            self.assertEqual(page_meta(total=1000, page=1, per_page=n)["per_page"], n)


class Paginate(unittest.TestCase):
    def test_slices_the_requested_window(self):
        items = list(range(1, 101))                            # 1..100
        p = paginate(items, page=2, per_page=25)
        self.assertEqual(p["rows"], list(range(26, 51)))

    def test_a_page_number_past_the_end_returns_the_last_page_not_nothing(self):
        items = list(range(1, 11))                              # only one page exists at per_page=25
        p = paginate(items, page=5, per_page=25)
        self.assertEqual(p["rows"], items)

    def test_empty_list_returns_no_items_without_raising(self):
        p = paginate([], page=1, per_page=25)
        self.assertEqual(p["rows"], [])
        self.assertEqual(p["pages"], 1)

    def test_the_key_is_not_named_items_which_a_dict_shadows_with_a_bound_method(self):
        # regression test: Jinja's `.` on a dict tries getattr() before getitem(), so a key
        # named "items" would silently resolve to dict.items (a method) in a template instead
        # of this list — caught live via a local smoke test of arena.html, not by a unit test,
        # since unittest's plain `p["items"]` access doesn't go through Jinja's attribute lookup.
        p = paginate([1, 2, 3], page=1, per_page=25)
        self.assertNotIn("items", p)
        self.assertIsInstance(p["rows"], list)


class PagerLinks(unittest.TestCase):
    def test_first_page_has_no_prev_link(self):
        meta = page_meta(total=100, page=1, per_page=25)
        links = pager_links({}, "page", "per_page", meta)
        self.assertIsNone(links["prev"])
        self.assertIsNotNone(links["next"])

    def test_last_page_has_no_next_link(self):
        meta = page_meta(total=100, page=4, per_page=25)          # 100/25 = exactly 4 pages
        links = pager_links({}, "page", "per_page", meta)
        self.assertIsNotNone(links["prev"])
        self.assertIsNone(links["next"])

    def test_middle_page_has_both(self):
        meta = page_meta(total=100, page=2, per_page=25)
        links = pager_links({}, "page", "per_page", meta)
        self.assertIsNotNone(links["prev"])
        self.assertIsNotNone(links["next"])

    def test_size_options_cover_every_page_size_and_flag_the_current_one(self):
        meta = page_meta(total=100, page=2, per_page=25)
        links = pager_links({}, "page", "per_page", meta)
        self.assertEqual([s["n"] for s in links["sizes"]], list(PAGE_SIZES))
        self.assertEqual([s["on"] for s in links["sizes"]], [n == 25 for n in PAGE_SIZES])

    def test_changing_page_size_resets_to_page_one(self):
        meta = page_meta(total=100, page=3, per_page=25)
        links = pager_links({}, "page", "per_page", meta)
        size_50 = next(s for s in links["sizes"] if s["n"] == 50)
        self.assertIn("page=1", size_50["href"])

    def test_other_query_params_survive_pagination(self):
        meta = page_meta(total=100, page=1, per_page=25)
        links = pager_links({"sort": "return_pct"}, "page", "per_page", meta)
        self.assertIn("sort=return_pct", links["next"])

    def test_flash_message_does_not_survive_a_pagination_click(self):
        # msg is a one-shot flash from a redirect (see app/main.py's back()); carrying it into a
        # pager link would make it reappear every time that list is paginated, stale.
        meta = page_meta(total=100, page=1, per_page=25)
        links = pager_links({"msg": "Import: 718 added."}, "page", "per_page", meta)
        self.assertNotIn("msg=", links["next"])

    def test_a_second_paginated_lists_params_are_left_untouched(self):
        # wallet_detail.html has two independent pagers on one page: the trade log (page/
        # per_page) and closed round-trips (cpage/cper_page). Paginating one must not touch
        # the other's current position.
        meta = page_meta(total=100, page=1, per_page=25)
        links = pager_links({"cpage": "3", "cper_page": "50"}, "page", "per_page", meta)
        self.assertIn("cpage=3", links["next"])
        self.assertIn("cper_page=50", links["next"])

    def test_stale_page_and_size_in_the_url_are_overwritten_not_duplicated(self):
        # the incoming URL already has page=2&per_page=25; the size links must switch per_page,
        # not just append a second copy of it.
        meta = page_meta(total=100, page=2, per_page=25)
        links = pager_links({"page": "2", "per_page": "25"}, "page", "per_page", meta)
        size_100 = next(s for s in links["sizes"] if s["n"] == 100)
        self.assertEqual(size_100["href"].count("per_page="), 1)
        self.assertIn("per_page=100", size_100["href"])
