"""Paper futures engine + background scheduler.

Runs any number of independent paper bots (see app/strategy_store.py): each has its own
strategy config, account, trades, kill switch and event log, identified by `bid` (bot id).

PAPER ONLY: this module reads public prices and records simulated trades in
SQLite. It has no way to place a real order.

Pessimistic choices (so paper results are not flattering):
  * Every entry and every market exit fills `slippage_pct` worse than the price.
  * Every fill pays the taker fee.
  * Stops/targets are checked against each 15m candle's HIGH and LOW, so wicks
    count. If one candle touches both stop and target, the STOP wins.
  * A stop that gaps through fills at the worse of stop and the candle open.
  * Funding is charged for the whole holding time at the current |rate|,
    whether it would have been paid or received.
"""
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from app import strategy_store as store
from app.db import db
from app.exchange import bitget
from app.strategies import get_strategy

_LOCKS: dict[int, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def bot_lock(bid: int) -> threading.RLock:
    """One lock per bot, so a manual 'Scan now' can't overlap that bot's scheduled scan."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(bid, threading.RLock())


# ── small helpers ────────────────────────────────────────────────────────────

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def log_event(bid: int, level: str, message: str) -> None:
    with db() as c:
        c.execute("INSERT INTO events(ts, source, level, message, bot_id) VALUES(?,?,?,?,?)",
                  (iso(utcnow()), "futures", level, message, bid))
        c.execute("DELETE FROM events WHERE bot_id=? AND id NOT IN "
                  "(SELECT id FROM events WHERE bot_id=? ORDER BY id DESC LIMIT 500)", (bid, bid))


def get_state(bid: int, key: str, default=None):
    with db() as c:
        row = c.execute("SELECT value FROM bot_state WHERE bot_id=? AND key=?", (bid, key)).fetchone()
    return row["value"] if row else default


def set_state(bid: int, key: str, value) -> None:
    with db() as c:
        c.execute("INSERT INTO bot_state(bot_id, key, value) VALUES(?,?,?) "
                  "ON CONFLICT(bot_id, key) DO UPDATE SET value=excluded.value", (bid, key, str(value)))


def get_balance(bid: int, cfg: dict) -> float:
    v = get_state(bid, "balance")
    if v is None:
        set_state(bid, "balance", cfg["starting_capital"])
        return float(cfg["starting_capital"])
    return float(v)


def closed_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Drop the still-forming candle. Bitget includes it as the last row, and
    acting on it is lookahead: its close changes until the period ends."""
    if df.empty:
        return df
    tf = pd.Timedelta(seconds=bitget.TF_SECONDS[timeframe])
    return df[df["ts"] + tf <= pd.Timestamp(utcnow())].reset_index(drop=True)


def _history(bid: int, sym: str, timeframe: str) -> pd.DataFrame:
    """All stored CLOSED candles for `sym`, topped up first (usually one small request).

    The live bot computes its signals on exactly the same stored history the backtester
    uses, so a signal here is the signal the backtest would have seen. That matters for
    long indicators: a 200-period EMA computed from only a few hundred recent candles
    gives different values from the backtest's, and the exchange's live-candle endpoint
    only returns ~90 days anyway. If the top-up fails we carry on with what is stored.
    """
    from app.backtest import data as btdata
    try:
        btdata.update_candles(sym, timeframe)
    except Exception as e:
        log_event(bid, "warn", f"Could not refresh {sym} {timeframe} candles ({type(e).__name__}); using stored data")
    return btdata.load_candles(sym, timeframe)


# ── exits ────────────────────────────────────────────────────────────────────

def _decide_exit(bid: int, t, cfg: dict, strat, params: dict):
    """Returns (price, outcome, when) or None. `t` is an open trade row."""
    slip = cfg["slippage_pct"] / 100
    entry_dt = datetime.fromisoformat(t["entry_time"])
    sym = t["symbol"]

    # 1) Walk 15m candles since entry, oldest first, so wicks are honoured. The history
    #    endpoint is used so trades held longer than ~5 days are still fully covered.
    try:
        df = bitget.get_history_candles(sym, "15m", int(entry_dt.timestamp() * 1000),
                                        int(utcnow().timestamp() * 1000) + 900_000)
    except Exception:
        return None                       # no data: do nothing this scan
    if df.empty:
        return None
    df = df[df["ts"] > pd.Timestamp(entry_dt)]
    for _, k in df.iterrows():
        when = k["ts"].to_pydatetime() + timedelta(minutes=15)
        if k["low"] <= t["stop"]:
            fill = min(t["stop"], k["open"]) * (1 - slip)
            return fill, "STOP LOSS", when
        if k["high"] >= t["target"]:
            return t["target"], "TAKE PROFIT", when

    # 2) Time stop.
    px = bitget.get_ticker(sym)
    if px is None:
        return None
    if utcnow() - entry_dt >= timedelta(hours=cfg["max_hold_hours"]):
        return px * (1 - slip), "MAX HOLD", utcnow()

    # 3) Strategy's own exit rule (e.g. RSI too high).
    tf_df = _history(bid, sym, cfg["timeframe"])
    if not tf_df.empty and strat.exit_signal(tf_df, params):
        return px * (1 - slip), "STRATEGY EXIT", utcnow()
    return None


def _close_trade(bid: int, t, cfg: dict, price: float, outcome: str, when: datetime) -> None:
    entry_dt = datetime.fromisoformat(t["entry_time"])
    hours = max((when - entry_dt).total_seconds() / 3600, 0)
    rate = bitget.get_funding_rate(t["symbol"])
    rate = abs(rate) if rate is not None else 0.0001     # fall back to a typical rate
    funding = t["notional"] * rate * hours / 8

    gross = (price - t["entry_price"]) * t["units"]
    fee_close = price * t["units"] * cfg["taker_fee"]
    # fee_open was already deducted from the balance when the trade opened.
    net = gross - t["fee_open"] - fee_close - funding
    balance = get_balance(bid, cfg) + gross - fee_close - funding
    set_state(bid, "balance", balance)

    with db() as c:
        c.execute(
            "UPDATE futures_trades SET status='closed', exit_time=?, exit_price=?, outcome=?, "
            "gross_pnl=?, fee_close=?, funding=?, net_pnl=?, balance_after=? WHERE id=?",
            (iso(when), price, outcome, gross, fee_close, funding, net, balance, t["id"]),
        )
    log_event(bid, "info", f"CLOSE {t['symbol']} {outcome}  net ${net:+.3f}  balance ${balance:.2f}")


def _manage_open(bid: int, cfg: dict, strat, params: dict) -> None:
    with db() as c:
        rows = c.execute("SELECT * FROM futures_trades WHERE status='open' AND bot_id=?", (bid,)).fetchall()
    for t in rows:
        decision = _decide_exit(bid, t, cfg, strat, params)
        px = bitget.get_ticker(t["symbol"])
        if px:
            set_state(bid, f"mark:{t['symbol']}", px)
        if decision:
            price, outcome, when = decision
            _close_trade(bid, t, cfg, price, outcome, when)


# ── entries ──────────────────────────────────────────────────────────────────

def _look_for_entries(bid: int, cfg: dict, strat, params: dict) -> None:
    with db() as c:
        open_rows = c.execute("SELECT symbol, margin FROM futures_trades WHERE status='open' AND bot_id=?",
                              (bid,)).fetchall()
    open_syms = {r["symbol"] for r in open_rows}
    used_margin = sum(r["margin"] for r in open_rows)
    slip = cfg["slippage_pct"] / 100

    for sym in cfg["pairs"]:
        if len(open_syms) >= cfg["max_simultaneous"]:
            break
        if sym in open_syms:
            continue

        df = _history(bid, sym, cfg["timeframe"])
        if df.empty:
            continue
        sig = strat.signal(sym, df, params)
        if sig is None:
            continue

        # Each signal candle may only be traded once. Without this, a trade that
        # stops out quickly would re-enter on the same candle 15 minutes later.
        if get_state(bid, f"last_signal:{sym}") == sig.candle_time:
            continue
        set_state(bid, f"last_signal:{sym}", sig.candle_time)

        fr = bitget.get_funding_rate(sym)
        if fr is not None and abs(fr) > cfg["max_funding_abs"]:
            log_event(bid, "warn", f"Skipped {sym}: funding {fr * 100:.3f}% too high")
            continue
        px = bitget.get_ticker(sym)
        if px is None:
            continue

        fill = px * (1 + slip)                       # pay slippage on entry
        if fill <= sig.stop or fill >= sig.target:
            log_event(bid, "info", f"Skipped {sym}: price already outside stop/target range")
            continue

        balance = get_balance(bid, cfg)
        risk_per_unit = fill - sig.stop
        units = balance * cfg["risk_pct"] / 100 / risk_per_unit
        notional = units * fill
        margin = notional / cfg["leverage"]
        fee_open = notional * cfg["taker_fee"]
        available = balance - used_margin
        if margin + fee_open > min(balance * cfg["max_margin_pct"] / 100, available):
            log_event(bid, "warn", f"Skipped {sym}: needs ${margin:.2f} margin, over the limit")
            continue

        set_state(bid, "balance", balance - fee_open)     # opening fee is paid immediately
        with db() as c:
            c.execute(
                "INSERT INTO futures_trades(symbol, strategy, status, entry_time, entry_price, "
                "stop, target, units, notional, margin, fee_open, signal_time, bot_id) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sym, strat.name, "open", iso(utcnow()), fill, sig.stop, sig.target,
                 units, notional, margin, fee_open, sig.candle_time, bid),
            )
        open_syms.add(sym)
        used_margin += margin
        log_event(bid, "info", f"OPEN {sym} @ {fill:.4f}  stop {sig.stop:.4f}  target {sig.target:.4f}  "
                          f"notional ${notional:.2f}  rsi {sig.rsi:.1f}")


# ── risk switch ──────────────────────────────────────────────────────────────

def _check_kill_switch(bid: int, cfg: dict) -> bool:
    """Halts trading if today's loss (vs the balance at the start of the UTC day)
    exceeds the daily limit. Returns True when halted."""
    today = utcnow().date().isoformat()
    balance = get_balance(bid, cfg)
    if get_state(bid, "day") != today:
        set_state(bid, "day", today)
        set_state(bid, "day_start_balance", balance)
    start = float(get_state(bid, "day_start_balance", balance))
    if start > 0 and (balance - start) / start * 100 <= -cfg["daily_loss_limit_pct"]:
        set_state(bid, "halted", f"Daily loss limit hit ({(balance - start) / start * 100:.1f}%)")
        log_event(bid, "error", "KILL SWITCH: " + get_state(bid, "halted"))
    return bool(get_state(bid, "halted"))


def resume(bid: int) -> None:
    set_state(bid, "halted", "")
    log_event(bid, "info", "Kill switch cleared manually")


# ── one scan, and the background loop ────────────────────────────────────────

def scan_once(bid: int) -> str:
    """One full pass for one bot. Safe to call from the web UI ('Scan now')."""
    with bot_lock(bid):
        bot = store.get_bot(bid)
        if bot is None:
            return "missing"
        if bot["status"] != "running":
            return "paused"
        cfg = bot["cfg"]
        try:
            strat = get_strategy(cfg["strategy"])
            params = cfg["params"]
            get_balance(bid, cfg)
            _manage_open(bid, cfg, strat, params)
            if _check_kill_switch(bid, cfg):
                set_state(bid, "last_scan", iso(utcnow()))
                return "halted"
            _look_for_entries(bid, cfg, strat, params)
            set_state(bid, "last_scan", iso(utcnow()))
            set_state(bid, "last_error", "")
            return "ok"
        except Exception as e:                      # keep the loop alive no matter what
            set_state(bid, "last_error", f"{type(e).__name__}: {e}")
            log_event(bid, "error", f"Scan failed: {type(e).__name__}: {e}")
            return "error"


def reset_account(bid: int) -> None:
    """Wipe this bot's paper trades and start over from its starting capital."""
    with bot_lock(bid):
        bot = store.get_bot(bid)
        with db() as c:
            c.execute("DELETE FROM futures_trades WHERE bot_id=?", (bid,))
            c.execute("DELETE FROM bot_state WHERE bot_id=?", (bid,))
            c.execute("UPDATE bots SET started_at=? WHERE id=?", (store.now_iso(), bid))
        set_state(bid, "balance", bot["cfg"]["starting_capital"])
        log_event(bid, "info", f"Paper account reset to ${bot['cfg']['starting_capital']:.2f}")


def scan_due(bot: dict) -> bool:
    last = get_state(bot["id"], "last_scan")
    if not last:
        return True
    dt = datetime.fromisoformat(last)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (utcnow() - dt).total_seconds() >= max(int(bot["cfg"]["scan_minutes"]), 1) * 60


class Runner:
    """One background thread; every TICK seconds it scans each running bot whose own
    scan interval has elapsed. Bots are independent: one failing never stops the others."""
    TICK = 20

    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="futures-runner", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        self._stop.wait(5)                          # let the web server come up first
        while not self._stop.is_set():
            try:
                for bot in store.list_bots("running"):
                    if self._stop.is_set():
                        break
                    if scan_due(bot):
                        scan_once(bot["id"])
            except Exception:                       # never let the loop die
                pass
            self._stop.wait(self.TICK)


runner = Runner()


# ── read-side helpers for the web pages ──────────────────────────────────────

def summary(bid: int, cfg: dict) -> dict:
    with db() as c:
        closed = c.execute("SELECT * FROM futures_trades WHERE status='closed' AND bot_id=? "
                           "ORDER BY exit_time", (bid,)).fetchall()
    balance = float(get_state(bid, "balance", cfg["starting_capital"]))
    start = float(cfg["starting_capital"])
    wins = [t["net_pnl"] for t in closed if t["net_pnl"] > 0]
    losses = [t["net_pnl"] for t in closed if t["net_pnl"] <= 0]

    equity, peak, max_dd = [start], start, 0.0
    for t in closed:
        equity.append(t["balance_after"])
        peak = max(peak, t["balance_after"])
        max_dd = min(max_dd, (t["balance_after"] - peak) / peak * 100)

    return {
        "balance": balance,
        "return_pct": (balance - start) / start * 100 if start else 0.0,
        "trades": len(closed),
        "win_rate": len(wins) / len(closed) * 100 if closed else 0.0,
        "profit_factor": (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else None,
        "max_drawdown": max_dd,
        "fees": sum((t["fee_open"] or 0) + (t["fee_close"] or 0) for t in closed),
        "funding": sum(t["funding"] or 0 for t in closed),
        "equity": equity,
        "equity_labels": ["start"] + [t["exit_time"][:16].replace("T", " ") for t in closed],
    }


_HISTORY_CACHE: dict = {}


def signal_history(cfg: dict) -> dict | None:
    """How often the CURRENT settings would have produced an entry signal over the stored history.

    Answers "will this bot ever trade?" without waiting weeks to find out. None when no
    history has been downloaded yet. Cached until the settings or the data change.
    """
    from app.backtest import data as btdata          # local import: the backtester is optional here

    strat = get_strategy(cfg["strategy"])
    params = cfg["params"]
    cov = [c for c in btdata.coverage() if c["tf"] == cfg["timeframe"] and c["symbol"] in cfg["pairs"]]
    if not cov:
        return None
    key = (tuple(sorted(cfg["pairs"])), cfg["timeframe"], cfg["strategy"],
           json.dumps(params, sort_keys=True), max(c["last_ms"] for c in cov))
    if key not in _HISTORY_CACHE:
        total, years = 0, 0.0
        for c in cov:
            df = btdata.load_candles(c["symbol"], c["tf"])
            total += int(strat.compute(df, params)["enter"].sum())
            years = max(years, (c["last_ms"] - c["first_ms"]) / (365.25 * 86_400_000))
        _HISTORY_CACHE.clear()
        _HISTORY_CACHE[key] = {"signals": total, "pairs": len(cov), "years": years,
                               "per_pair_year": total / (years * len(cov)) if years else 0.0}
    return _HISTORY_CACHE[key]
