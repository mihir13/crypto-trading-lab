"""Historical data for backtests: download once, keep in SQLite, load as arrays.

Only CLOSED candles are stored. A candle that is still forming has a close price
that keeps changing, so using it would be lookahead.
"""
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from app.db import db
from app.exchange import bitget

TF_MS = {k: v * 1000 for k, v in bitget.TF_SECONDS.items()}
PAGE_PAUSE = 0.12          # seconds between requests; stays far below Bitget's rate limit


def _now_ms() -> int:
    return int(time.time() * 1000)


def _get(path: str, params: dict) -> list:
    """GET with a couple of retries; public endpoints only."""
    last = None
    for attempt in range(4):
        try:
            data = bitget._get(path, params, timeout=20)
            return data["data"]
        except (requests.RequestException, RuntimeError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Bitget request failed after retries: {last}")


# ── download ─────────────────────────────────────────────────────────────────

def update_candles(symbol: str, tf: str, progress=None) -> int:
    """Fetch candles newer than what we have (or the whole history the first time).

    Walks BACKWARDS from now in pages of 200 until it reaches candles we already
    stored, or the exchange has nothing older. Returns the number of new rows.
    """
    tf_ms = TF_MS[tf]
    with db() as c:
        row = c.execute("SELECT MAX(ts) m FROM candles WHERE symbol=? AND tf=?", (symbol, tf)).fetchone()
    have_until = row["m"]            # None on first download
    end = _now_ms()
    added, pages = 0, 0
    while True:
        rows = _get("/api/v2/mix/market/history-candles", {
            "symbol": symbol, "productType": bitget.PRODUCT,
            "granularity": bitget._GRAN[tf], "endTime": str(end), "limit": "200"})
        if not rows:
            break
        pages += 1
        closed = [r for r in rows if int(r[0]) + tf_ms <= _now_ms()]
        batch = [(symbol, tf, int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                 for r in closed]
        with db() as c:
            before = c.execute("SELECT COUNT(*) n FROM candles WHERE symbol=? AND tf=?", (symbol, tf)).fetchone()["n"]
            c.executemany("INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?)", batch)
            after = c.execute("SELECT COUNT(*) n FROM candles WHERE symbol=? AND tf=?", (symbol, tf)).fetchone()["n"]
        added += after - before
        oldest = min(int(r[0]) for r in rows)
        if progress:
            progress(f"{symbol} {tf}: candles back to {datetime.fromtimestamp(oldest / 1000, timezone.utc):%Y-%m-%d}")
        if have_until is not None and oldest <= have_until:
            break                     # reached data we already had
        if oldest >= end:
            break                     # no older data
        end = oldest
        time.sleep(PAGE_PAUSE)
        if pages > 2000:              # safety stop
            break
    return added


def update_funding(symbol: str, progress=None) -> int:
    """Fetch historical funding rates (newest first, 100 per page)."""
    with db() as c:
        row = c.execute("SELECT MAX(ts) m FROM funding WHERE symbol=?", (symbol,)).fetchone()
    have_until = row["m"]
    added, page = 0, 1
    while page < 400:
        rows = _get("/api/v2/mix/market/history-fund-rate", {
            "symbol": symbol, "productType": bitget.PRODUCT, "pageSize": "100", "pageNo": str(page)})
        if not rows:
            break
        batch = [(symbol, int(r["fundingTime"]), float(r["fundingRate"]), "bitget") for r in rows]
        with db() as c:
            before = c.execute("SELECT COUNT(*) n FROM funding WHERE symbol=?", (symbol,)).fetchone()["n"]
            c.executemany("INSERT OR REPLACE INTO funding VALUES (?,?,?,?)", batch)   # real data wins
            after = c.execute("SELECT COUNT(*) n FROM funding WHERE symbol=?", (symbol,)).fetchone()["n"]
        added += after - before
        oldest = min(b[1] for b in batch)
        if progress:
            progress(f"{symbol} funding back to {datetime.fromtimestamp(oldest / 1000, timezone.utc):%Y-%m-%d}")
        if have_until is not None and oldest <= have_until:
            break
        page += 1
        time.sleep(PAGE_PAUSE)
    return added


def update_funding_proxy(symbol: str, progress=None) -> int:
    """Older funding history from Binance's public API (Bitget only keeps ~90 days).

    This is a PROXY: same 8-hour mechanism and usually close, but not Bitget's own
    numbers. Rows never overwrite Bitget's real ones (INSERT OR IGNORE), and each
    row is tagged src='binance' so the UI can say so.
    """
    with db() as c:
        row = c.execute("SELECT MAX(ts) m FROM funding WHERE symbol=? AND src='binance'", (symbol,)).fetchone()
    start = (row["m"] + 1) if row["m"] else int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    added = 0
    while start < _now_ms():
        last = None
        for attempt in range(4):
            try:
                r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                                 params={"symbol": symbol, "startTime": start, "limit": 1000}, timeout=20)
                r.raise_for_status()
                rows = r.json()
                break
            except (requests.RequestException, ValueError) as e:
                last, rows = e, None
                time.sleep(1.5 * (attempt + 1))
        if rows is None:
            raise RuntimeError(f"Binance funding request failed: {last}")
        if not rows:
            break
        # Binance stamps rows a few ms after the funding time; round to the minute so they
        # line up with Bitget's timestamps.
        batch = [(symbol, round(int(x["fundingTime"]) / 60000) * 60000, float(x["fundingRate"]), "binance")
                 for x in rows]
        with db() as c:
            before = c.execute("SELECT COUNT(*) n FROM funding WHERE symbol=?", (symbol,)).fetchone()["n"]
            c.executemany("INSERT OR IGNORE INTO funding VALUES (?,?,?,?)", batch)
            after = c.execute("SELECT COUNT(*) n FROM funding WHERE symbol=?", (symbol,)).fetchone()["n"]
        added += after - before
        newest = int(rows[-1]["fundingTime"])
        if progress:
            progress(f"{symbol} funding (Binance proxy) up to {datetime.fromtimestamp(newest / 1000, timezone.utc):%Y-%m-%d}")
        if len(rows) < 1000:
            break
        start = newest + 1
        time.sleep(PAGE_PAUSE)
    return added


# ── read ─────────────────────────────────────────────────────────────────────

def load_candles(symbol: str, tf: str) -> pd.DataFrame:
    with db() as c:
        rows = c.execute("SELECT ts, open, high, low, close, vol FROM candles "
                         "WHERE symbol=? AND tf=? ORDER BY ts", (symbol, tf)).fetchall()
    df = pd.DataFrame([tuple(r) for r in rows], columns=["ts_ms", "open", "high", "low", "close", "vol"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df


def load_funding(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    with db() as c:
        rows = c.execute("SELECT ts, rate FROM funding WHERE symbol=? ORDER BY ts", (symbol,)).fetchall()
    return (np.array([r["ts"] for r in rows], dtype=np.int64),
            np.array([r["rate"] for r in rows], dtype=float))


def coverage() -> list[dict]:
    """What we have stored, per symbol and timeframe, including any missing candles."""
    out = []
    with db() as c:
        rows = c.execute("SELECT symbol, tf, MIN(ts) a, MAX(ts) b, COUNT(*) n FROM candles "
                         "GROUP BY symbol, tf ORDER BY symbol, tf").fetchall()
        fund = {r["symbol"]: (r["n"], r["a"], r["b"], r["real"]) for r in c.execute(
            "SELECT symbol, COUNT(*) n, MIN(ts) a, MAX(ts) b, SUM(src='bitget') real "
            "FROM funding GROUP BY symbol").fetchall()}
    for r in rows:
        expected = (r["b"] - r["a"]) // TF_MS[r["tf"]] + 1
        f = fund.get(r["symbol"])
        out.append({
            "symbol": r["symbol"], "tf": r["tf"], "first_ms": r["a"], "last_ms": r["b"], "candles": r["n"],
            "missing": max(expected - r["n"], 0),
            "first": datetime.fromtimestamp(r["a"] / 1000, timezone.utc).strftime("%Y-%m-%d"),
            "last": datetime.fromtimestamp(r["b"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "funding_rows": f[0] if f else 0,
            "funding_first": datetime.fromtimestamp(f[1] / 1000, timezone.utc).strftime("%Y-%m-%d") if f else "",
            "funding_real": f[3] if f else 0,          # rows from Bitget itself (the rest is proxy)
        })
    return out
