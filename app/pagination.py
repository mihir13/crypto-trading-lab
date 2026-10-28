"""Pagination shared by every list page (currently a live bot's closed trades). One tiny
module so every list behaves the same way (same page sizes, same URL params, same "X-Y of N"
wording) and no route hand-rolls its own clamping or link building.
"""
from urllib.parse import urlencode

PAGE_SIZES = (10, 25, 50, 100)
DEFAULT_PAGE_SIZE = 25


def page_meta(total: int, page: int, per_page: int) -> dict:
    """Clamp `page`/`per_page` into range for `total` items. Returns everything a route or
    template needs — `offset` for a SQL LIMIT/OFFSET or a Python slice, `start`/`end` as the
    1-indexed bounds to show as "X-Y of N" — but does not slice anything itself: one caller wants
    a SQL OFFSET and another wants a Python list slice (see `paginate` for the latter).
    """
    per_page = per_page if per_page in PAGE_SIZES else DEFAULT_PAGE_SIZE
    pages = max(-(-total // per_page), 1)          # ceil division; at least 1 page even when empty
    page = min(max(page, 1), pages)
    offset = (page - 1) * per_page
    return {
        "page": page, "per_page": per_page, "pages": pages, "total": total, "offset": offset,
        "start": (offset + 1) if total else 0, "end": min(offset + per_page, total),
    }


def paginate(items: list, page: int, per_page: int) -> dict:
    """For a list that's already fully computed in Python — e.g. a ranked leaderboard's rows, which all
    have to be scored before they can be ranked and one page of them picked. Prefer a direct SQL
    LIMIT/OFFSET query instead (with `page_meta` for the clamping) when the full list doesn't
    otherwise need to exist just to show one page of it.
    """
    # NOT "items": Jinja's `.` on a dict tries getattr() before getitem(), and dict.items is a
    # real bound method — a key with that name would silently resolve to the method instead of
    # this list in a template (the same class of bug the "copy"/dict.copy() collision was).
    meta = page_meta(len(items), page, per_page)
    meta["rows"] = items[meta["offset"]: meta["offset"] + meta["per_page"]]
    return meta


def pager_links(current_query: dict, page_param: str, size_param: str, meta: dict) -> dict:
    """Hrefs for the pager control: prev/next and each page-size option, preserving every other
    query param already on the URL — so a second, independently paginated list on the same page
    (identified by its own `page_param`/`size_param`, e.g. a trade log vs. a second list) isn't disturbed — except a one-shot flash `msg`, which shouldn't keep reappearing
    every time this list is paginated.
    """
    base = {k: str(v) for k, v in current_query.items() if k != "msg"}

    def href(page=None, per_page=None) -> str:
        q = dict(base)
        q[page_param] = str(page if page is not None else meta["page"])
        q[size_param] = str(per_page if per_page is not None else meta["per_page"])
        return "?" + urlencode(q)

    return {
        "prev": href(page=meta["page"] - 1) if meta["page"] > 1 else None,
        "next": href(page=meta["page"] + 1) if meta["page"] < meta["pages"] else None,
        "sizes": [{"n": n, "href": href(page=1, per_page=n), "on": n == meta["per_page"]} for n in PAGE_SIZES],
    }
