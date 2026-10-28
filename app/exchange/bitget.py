"""Bitget PUBLIC market data. No API key is used or needed.

This module only reads prices. There is deliberately no order-placing code
anywhere in this project.
"""
import time

import pandas as pd
import requests

BASE_URL = "https://api.bitget.com"
PRODUCT = "USDT-FUTURES"

# Bitget wants "1H"/"4H" (capital H) but "15m" (lowercase m).
_GRAN = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
         "1h": "1H", "4h": "4H", "6h": "6H", "12h": "12H", "1d": "1D"}

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
              "1h": 3600, "4h": 14400, "6h": 21600, "12h": 43200, "1d": 86400}


def _get(path: str, params: dict, timeout: int = 10) -> dict:
    r = requests.get(BASE_URL + path, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "00000":
        raise RuntimeError(f"Bitget error: {data.get('msg')}")
    return data


def _to_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "turnover"])
    for col in ["open", "high", "low", "close", "vol", "turnover"]:
        df[col] = pd.to_numeric(df[col])
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def get_candles(symbol: str, timeframe: str = "4h", limit: int = 200) -> pd.DataFrame:
    """Most recent candles. The LAST row is usually the still-forming candle."""
    data = _get("/api/v2/mix/market/candles", {
        "symbol": symbol, "productType": PRODUCT,
        "granularity": _GRAN[timeframe.lower()], "limit": str(limit),
    })
    return _to_frame(data["data"])


def get_history_candles(symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Older candles between two timestamps (used by the backtester).

    Bitget returns at most 200 rows per call, so we walk backwards from
    end_ms until we pass start_ms.
    """
    frames = []
    cursor = end_ms
    while cursor > start_ms:
        data = _get("/api/v2/mix/market/history-candles", {
            "symbol": symbol, "productType": PRODUCT,
            "granularity": _GRAN[timeframe.lower()],
            "endTime": str(cursor), "limit": "200",
        })
        rows = data["data"]
        if not rows:
            break
        frame = _to_frame(rows)
        frames.append(frame)
        oldest = int(frame["ts"].iloc[0].timestamp() * 1000)
        if oldest >= cursor:
            break
        cursor = oldest
        time.sleep(0.15)  # stay well under the rate limit
    if not frames:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol", "turnover"])
    df = pd.concat(frames).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df[df["ts"] >= pd.to_datetime(start_ms, unit="ms", utc=True)].reset_index(drop=True)


def get_ticker(symbol: str) -> float | None:
    try:
        data = _get("/api/v2/mix/market/ticker", {"symbol": symbol, "productType": PRODUCT}, 8)
        return float(data["data"][0]["lastPr"])
    except Exception:
        return None


def get_funding_rate(symbol: str) -> float | None:
    """Current funding rate per 8h interval, or None if the request failed."""
    try:
        data = _get("/api/v2/mix/market/current-fund-rate",
                    {"symbol": symbol, "productType": PRODUCT}, 8)
        return float(data["data"][0]["fundingRate"])
    except Exception:
        return None


def parse_tickers(rows: list, symbols: list[str]) -> list[dict]:
    """[{symbol, price, change_pct}] for the wanted symbols, in the order asked (missing ones skipped)."""
    by = {r.get("symbol"): r for r in rows}
    out = []
    for s in symbols:
        r = by.get(s)
        if not r or not r.get("lastPr"):
            continue
        ch = r.get("change24h")
        out.append({"symbol": s, "price": float(r["lastPr"]),
                    "change_pct": float(ch) * 100 if ch not in (None, "") else None})
    return out


def get_tickers(symbols: list[str]) -> list[dict]:
    """Last price and 24h change for several symbols in one request (for the site's price strip)."""
    data = _get("/api/v2/mix/market/tickers", {"productType": PRODUCT})
    return parse_tickers(data["data"], symbols)


def parse_symbol_list(rows: list) -> list[dict]:
    """Every USDT perpetual as {symbol, base, volume, price, change_pct}, busiest first."""
    out = []
    for r in rows:
        sym = r.get("symbol", "")
        if not sym.endswith("USDT") or len(sym) <= 4:
            continue
        try:
            price, volume = float(r["lastPr"]), float(r.get("usdtVolume") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        ch = r.get("change24h")
        out.append({"symbol": sym, "base": sym[:-4], "volume": volume, "price": price,
                    "change_pct": float(ch) * 100 if ch not in (None, "") else None})
    out.sort(key=lambda x: -x["volume"])
    return out


def list_symbols() -> list[dict]:
    """All tradable USDT perpetuals (one request), for the pair picker and for checking a typed pair exists."""
    return parse_symbol_list(_get("/api/v2/mix/market/tickers", {"productType": PRODUCT})["data"])
