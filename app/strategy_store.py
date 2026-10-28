"""Saved strategies and the paper bots deployed from them.

A *strategy* is a saved configuration (type + parameters + pairs + timeframe + risk/cost
settings). A *bot* is one live paper deployment of a strategy with its own account. The bot
keeps a snapshot of the config it started with, so editing a strategy never changes a running
bot or rewrites what it has already done.
"""
import json
from datetime import datetime, timezone

from app.db import db
from app.strategies import STRATEGIES

RISK_DEFAULTS = {
    "starting_capital": 100.0,
    "risk_pct": 3.0,             # % of balance risked per trade
    "max_simultaneous": 3,
    "leverage": 5,
    "max_margin_pct": 40.0,      # max margin on one trade, % of balance
    "taker_fee": 0.0006,         # per side; every fill is assumed taker
    "slippage_pct": 0.05,        # applied against you on every market fill
    "max_hold_hours": 72,
    "max_funding_abs": 0.001,    # skip entries when |funding rate| is above this
    "daily_loss_limit_pct": 10.0,
    "scan_minutes": 15,
}
DEFAULT_PAIRS = ["SOLUSDT", "AVAXUSDT", "LINKUSDT"]
DEFAULT_TIMEFRAME = "4h"
TIMEFRAMES = ("1h", "4h", "1d")
MAX_PAIRS = 12                    # every pair is downloaded, backtested and scanned; keep it manageable


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── strategies ───────────────────────────────────────────────────────────────

def _strategy(row) -> dict:
    d = dict(row)
    d["params"] = {**STRATEGIES[d["type"]].default_params, **json.loads(d["params"])} \
        if d["type"] in STRATEGIES else json.loads(d["params"])
    d["pairs"] = json.loads(d["pairs"])
    d["risk"] = {**RISK_DEFAULTS, **json.loads(d["risk"])}
    return d


def list_strategies(include_archived: bool = False) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT * FROM strategies " + ("" if include_archived else "WHERE archived=0 ")
                         + "ORDER BY id").fetchall()
    return [_strategy(r) for r in rows]


def get_strategy(sid: int) -> dict | None:
    with db() as c:
        row = c.execute("SELECT * FROM strategies WHERE id=?", (sid,)).fetchone()
    return _strategy(row) if row else None


def save_strategy(fields: dict, sid: int | None = None) -> int:
    """Insert (sid None) or update. `fields`: name, type, params, pairs, timeframe, risk, notes."""
    vals = (fields["name"], fields["type"], json.dumps(fields["params"]), json.dumps(fields["pairs"]),
            fields["timeframe"], json.dumps(fields["risk"]), fields.get("notes", ""))
    with db() as c:
        if sid is None:
            cur = c.execute("INSERT INTO strategies(name, type, params, pairs, timeframe, risk, notes, created) "
                            "VALUES(?,?,?,?,?,?,?,?)", vals + (now_iso(),))
            return cur.lastrowid
        c.execute("UPDATE strategies SET name=?, type=?, params=?, pairs=?, timeframe=?, risk=?, notes=? "
                  "WHERE id=?", vals + (sid,))
    return sid


def set_archived(sid: int, archived: bool) -> None:
    with db() as c:
        c.execute("UPDATE strategies SET archived=? WHERE id=?", (int(archived), sid))


def strategy_config(s: dict) -> dict:
    """The flat config the engine and backtester work from: strategy type, params, pairs,
    timeframe and every risk/cost setting."""
    return {"strategy": s["type"], "params": dict(s["params"]), "pairs": list(s["pairs"]),
            "timeframe": s["timeframe"], **s["risk"]}


# ── bots ─────────────────────────────────────────────────────────────────────

def _bot(row) -> dict:
    d = dict(row)
    d["cfg"] = json.loads(d.pop("config"))
    return d


def create_bot(strategy_id: int, name: str | None = None) -> int:
    s = get_strategy(strategy_id)
    if s is None:
        raise KeyError(strategy_id)
    now = now_iso()
    with db() as c:
        cur = c.execute("INSERT INTO bots(strategy_id, name, config, status, created, started_at) "
                        "VALUES(?,?,?,?,?,?)",
                        (strategy_id, name or s["name"], json.dumps(strategy_config(s)), "running", now, now))
        bid = cur.lastrowid
        c.execute("INSERT INTO bot_state(bot_id, key, value) VALUES(?,?,?)",
                  (bid, "balance", str(s["risk"]["starting_capital"])))
        c.execute("INSERT INTO events(ts, source, level, message, bot_id) VALUES(?,?,?,?,?)",
                  (now, "futures", "info", f"Bot started from strategy '{s['name']}' with "
                                           f"${s['risk']['starting_capital']:.2f}", bid))
    return bid


def list_bots(status: str | None = None) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT * FROM bots " + ("WHERE status=? " if status else "") + "ORDER BY id",
                         (status,) if status else ()).fetchall()
    return [_bot(r) for r in rows]


def get_bot(bid: int) -> dict | None:
    with db() as c:
        row = c.execute("SELECT * FROM bots WHERE id=?", (bid,)).fetchone()
    return _bot(row) if row else None


def set_bot_status(bid: int, status: str) -> None:
    with db() as c:
        c.execute("UPDATE bots SET status=? WHERE id=?", (status, bid))


def bots_for_strategy(sid: int) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT * FROM bots WHERE strategy_id=? ORDER BY id", (sid,)).fetchall()
    return [_bot(r) for r in rows]


# ── one-time migration of the original single bot ────────────────────────────

def migrate_legacy() -> None:
    """Turn the original single global bot (futures.* / strategy.* settings, futures_state, and
    trades with no bot_id) into Strategy #1 + Bot #1. Runs once, only on an existing install."""
    with db() as c:
        if c.execute("SELECT COUNT(*) n FROM strategies").fetchone()["n"]:
            return
        legacy = {r["key"]: json.loads(r["value"]) for r in
                  c.execute("SELECT key, value FROM settings WHERE key LIKE 'futures.%' OR key LIKE 'strategy.%'")}
        untagged = c.execute("SELECT COUNT(*) n FROM futures_trades WHERE bot_id IS NULL").fetchone()["n"]
        if not legacy and not untagged:
            return                                            # fresh install: nothing to carry over
        stype = legacy.get("futures.strategy", "ema_rsi")
        if stype not in STRATEGIES:
            stype = "ema_rsi"
        strat = STRATEGIES[stype]
        params = {**strat.default_params, **legacy.get(f"strategy.{stype}", {})}
        risk = {k: legacy[f"futures.{k}"] for k in RISK_DEFAULTS if f"futures.{k}" in legacy}
        cur = c.execute(
            "INSERT INTO strategies(name, type, params, pairs, timeframe, risk, notes, created) VALUES(?,?,?,?,?,?,?,?)",
            (strat.label, stype, json.dumps(params), json.dumps(legacy.get("futures.pairs", DEFAULT_PAIRS)),
             legacy.get("futures.timeframe", DEFAULT_TIMEFRAME), json.dumps(risk),
             "Migrated from the original single paper bot.", now_iso()))
        sid = cur.lastrowid
        cfg = {"strategy": stype, "params": params, "pairs": legacy.get("futures.pairs", DEFAULT_PAIRS),
               "timeframe": legacy.get("futures.timeframe", DEFAULT_TIMEFRAME), **RISK_DEFAULTS, **risk}
        first = c.execute("SELECT MIN(entry_time) t FROM futures_trades").fetchone()["t"] or now_iso()
        status = "running" if legacy.get("futures.enabled", True) else "paused"
        bid = c.execute("INSERT INTO bots(strategy_id, name, config, status, created, started_at) VALUES(?,?,?,?,?,?)",
                        (sid, strat.label, json.dumps(cfg), status, now_iso(), first)).lastrowid
        c.execute("INSERT INTO bot_state(bot_id, key, value) SELECT ?, key, value FROM futures_state", (bid,))
        c.execute("UPDATE futures_trades SET bot_id=? WHERE bot_id IS NULL", (bid,))
        c.execute("UPDATE events SET bot_id=? WHERE bot_id IS NULL", (bid,))


def all_pairs() -> list[str]:
    """Every pair used by any saved strategy (plus the defaults): what 'Download data' fetches."""
    pairs = list(DEFAULT_PAIRS)
    for s in list_strategies():
        pairs += [p for p in s["pairs"] if p not in pairs]
    return pairs
