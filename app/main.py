"""Crypto Trading Lab: web app entry point.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8420

LAN-only by design (no login). Do not expose this port to the internet.
"""
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from app import db as database
from app import pagination as pg
from app import analysis, explain
from app import strategy_store as store
from app.backtest import data as btdata
from app.backtest import runs as btruns
from app.exchange import bitget
from app.futures import engine
from app.strategies import STRATEGIES

HERE = os.path.dirname(os.path.abspath(__file__))

# Every timestamp is stored in UTC (as it should be) but shown in UK local time — GMT in winter,
# BST in summer, switching on its own (a fixed offset would be wrong for half the year).
UK_TZ = ZoneInfo("Europe/London")


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init()
    store.migrate_legacy()
    with database.db() as c:
        c.execute("UPDATE bt_runs SET status='error', message='Interrupted by a restart' "
                  "WHERE status IN ('queued','running')")
    if os.environ.get("LAB_NO_RUNNER") != "1":     # handy for tests
        engine.runner.start()
    yield
    engine.runner.stop()


app = FastAPI(title="Crypto Trading Lab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))

NAV = [
    ("overview", "/", "Dashboard"),
    ("strategies", "/strategies", "Strategies"),
    ("bots", "/bots", "Live Bots"),
    ("backtest", "/backtest", "Backtest Lab"),
    ("analysis", "/analysis", "Analysis"),
]


def money(v, digits=2):
    return "–" if v is None else f"${v:,.{digits}f}"


def signed(v, digits=2):
    return "–" if v is None else f"{v:+,.{digits}f}"



def tv_url(symbol: str) -> str:
    """TradingView chart for a Bitget USDT perpetual, e.g. SOLUSDT -> BITGET:SOLUSDT.P."""
    return "https://www.tradingview.com/chart/?symbol=" + quote(f"BITGET:{symbol}.P", safe="") + "&theme=dark"


def pair(symbol: str, short: bool = True) -> Markup:
    """A pair as a link to its TradingView chart (opens in a new tab). `short` shows SOL, not SOLUSDT."""
    label = symbol.removesuffix("USDT") if short else symbol
    return Markup('<a class="pair" href="{}" target="_blank" rel="noopener noreferrer" title="Open {} on TradingView">{}</a>').format(
        tv_url(symbol), symbol, label)


def pair_list(symbols, sep: str = ", ", short: bool = True) -> Markup:
    return Markup(escape(sep)).join(pair(x, short) for x in symbols)


templates.env.filters["pair"] = pair
templates.env.filters["pair_list"] = pair_list
templates.env.filters["money"] = money
templates.env.filters["signed"] = signed


STATIC_DIR = os.path.join(HERE, "static")


def _asset_version() -> int:
    """Changes whenever the stylesheet changes, so browsers never show stale CSS."""
    return int(max(os.path.getmtime(os.path.join(STATIC_DIR, f)) for f in ("style.css", "pairs.js")))


def _bot_status() -> tuple[str, str]:
    """(dot colour class, text) for the little status chip in the sidebar."""
    bots = store.list_bots()
    running = [b for b in bots if b["status"] == "running"]
    halted = [b for b in running if engine.get_state(b["id"], "halted")]
    if halted:
        return "bad", f"{len(halted)} bot(s) halted"
    if not running:
        return "", "No bots running"
    return "good", f"{len(running)} bot{'s' if len(running) != 1 else ''} running"


def render(request: Request, name: str, active: str, **ctx):
    return templates.TemplateResponse(request, name, {
        "nav": NAV, "active": active,
        "bot_status": _bot_status(), "asset_v": _asset_version(), **ctx})


def back(path: str, msg: str = ""):
    url = path + (f"?msg={quote(msg)}" if msg else "")
    return RedirectResponse(url, status_code=303)


# ── Glossary ─────────────────────────────────────────────────────────────────

@app.get("/glossary")
def glossary_page(request: Request):
    return render(request, "glossary.html", "glossary", terms=[explain.define(k) for k in explain.GLOSSARY])


# ── Price strip (top of every page; filled in by a small script) ─────────────

_TICKER = {"at": 0.0, "rows": []}


@app.get("/api/ticker")
def api_ticker():
    """Last price and 24h change for every pair any strategy uses. Cached briefly so a few open
    tabs never hammer the exchange; on an error the previous prices are served."""
    now = time.time()
    if now - _TICKER["at"] > 15:
        _TICKER["at"] = now
        try:
            _TICKER["rows"] = bitget.get_tickers(store.all_pairs())
        except Exception:
            _TICKER["at"] = now - 10            # try again in ~5 s
    return _TICKER["rows"]


# ── Pair picker: every USDT perpetual Bitget lists (busiest first), cached ────────

_SYMBOLS = {"at": 0.0, "rows": []}


def market_symbols() -> list[dict]:
    """All Bitget USDT perpetuals. Cached for ten minutes (thirty seconds after a failed fetch);
    an empty list means the exchange couldn't be reached, and callers then skip the existence check."""
    now = time.time()
    if now - _SYMBOLS["at"] > (600 if _SYMBOLS["rows"] else 30):
        _SYMBOLS["at"] = now
        try:
            _SYMBOLS["rows"] = bitget.list_symbols()
        except Exception:
            pass
    return _SYMBOLS["rows"]


@app.get("/api/pairs")
def api_pairs():
    return market_symbols()


# ── Overview ─────────────────────────────────────────────────────────────────

def bot_card(bot: dict) -> dict:
    """Everything a list row/card needs about one bot."""
    summ = engine.summary(bot["id"], bot["cfg"])
    with database.db() as c:
        n_open = c.execute("SELECT COUNT(*) n FROM futures_trades WHERE status='open' AND bot_id=?",
                           (bot["id"],)).fetchone()["n"]
    return {**bot, "summ": summ, "n_open": n_open, "halted": engine.get_state(bot["id"], "halted"),
            "strat_label": STRATEGIES[bot["cfg"]["strategy"]].label,
            "last_scan": engine.get_state(bot["id"], "last_scan")}


@app.get("/")
def overview(request: Request):
    rows = [analysis.strategy_row(s) for s in store.list_strategies()]
    bots = [bot_card(b) for b in store.list_bots()]
    cov = btdata.coverage()
    alerts = []
    for b in bots:
        if b["halted"]:
            alerts.append(("bad", f"{b['name']} (bot #{b['id']}) is halted: {b['halted']}", f"/bots/{b['id']}"))
        elif engine.get_state(b["id"], "last_error") and b["status"] == "running":
            alerts.append(("warn", f"{b['name']} (bot #{b['id']}) had a scan error: {engine.get_state(b['id'], 'last_error')}",
                           f"/bots/{b['id']}"))
    job = _running_job()
    if job:
        alerts.append(("", f"A {job['kind']} job is running", f"/backtest/run/{job['id']}"))
    return render(request, "overview.html", "overview", rows=rows, bots=bots, alerts=alerts,
                  min_live=analysis.MIN_LIVE_TRADES,
                  live_trades=sum(b["summ"]["trades"] for b in bots),
                  data_through=max((c["last_ms"] for c in cov), default=None), n_series=len(cov))


# ── Live Bots (paper bots deployed from strategies; each has its own account) ────────

@app.get("/bots")
def bots_page(request: Request, msg: str = ""):
    cards = [bot_card(b) for b in store.list_bots()]
    cards.sort(key=lambda b: (b["status"] != "running", -b["id"]))
    return render(request, "bots.html", "bots", bots=cards, msg=msg)


@app.get("/futures")
def futures_redirect():
    """The old single Trading Bot page (bookmarks)."""
    return RedirectResponse("/bots", status_code=307)


@app.get("/bots/{bid}")
def bot_page(request: Request, bid: int, msg: str = "", page: int = 1, per_page: int = pg.DEFAULT_PAGE_SIZE):
    bot = store.get_bot(bid)
    if bot is None:
        return back("/bots", "That bot doesn't exist")
    cfg = bot["cfg"]
    with database.db() as c:
        open_rows = c.execute("SELECT * FROM futures_trades WHERE status='open' AND bot_id=? "
                              "ORDER BY entry_time DESC", (bid,)).fetchall()
        closed_total = c.execute("SELECT COUNT(*) n FROM futures_trades WHERE status='closed' AND bot_id=?",
                                 (bid,)).fetchone()["n"]
    pager = pg.page_meta(closed_total, page, per_page)
    with database.db() as c:
        closed = c.execute("SELECT * FROM futures_trades WHERE status='closed' AND bot_id=? "
                           "ORDER BY exit_time DESC LIMIT ? OFFSET ?",
                           (bid, pager["per_page"], pager["offset"])).fetchall()
        events = c.execute("SELECT * FROM events WHERE bot_id=? ORDER BY id DESC LIMIT 25", (bid,)).fetchall()
    pager["links"] = pg.pager_links(dict(request.query_params), "page", "per_page", pager)

    positions = []
    for t in open_rows:
        mark = engine.get_state(bid, f"mark:{t['symbol']}")
        mark = float(mark) if mark else None
        positions.append({**dict(t), "mark": mark,
                          "unrealized": (mark - t["entry_price"]) * t["units"] if mark else None})

    last_scan = engine.get_state(bid, "last_scan")
    next_scan = None
    if last_scan:
        last_scan_dt = datetime.fromisoformat(last_scan)
        if last_scan_dt.tzinfo is None:
            last_scan_dt = last_scan_dt.replace(tzinfo=timezone.utc)
        next_scan = ((last_scan_dt + timedelta(minutes=cfg["scan_minutes"]))
                     .astimezone(UK_TZ).strftime("%H:%M"))
    strat = STRATEGIES[cfg["strategy"]]
    return render(request, "bot.html", "bots", bot=bot, cfg=cfg, summ=engine.summary(bid, cfg),
                  positions=positions, closed=closed, pager=pager, events=events,
                  halted=engine.get_state(bid, "halted"),
                  last_scan=iso_local(last_scan) if last_scan else "never", next_scan=next_scan,
                  last_error=engine.get_state(bid, "last_error"), sig_hist=engine.signal_history(cfg),
                  describe=strat.describe(cfg["params"]),
                  strat_label=strat.label, evidence=strat.evidence, msg=msg)


@app.post("/bots/{bid}/scan")
def bot_scan(bid: int):
    return back(f"/bots/{bid}", f"Scan finished: {engine.scan_once(bid)}")


@app.post("/bots/{bid}/resume")
def bot_resume(bid: int):
    engine.resume(bid)
    return back(f"/bots/{bid}", "Kill switch cleared")


@app.post("/bots/{bid}/status")
def bot_set_status(bid: int, status: str = Form(...)):
    if status not in ("running", "paused") or store.get_bot(bid) is None:
        return back("/bots", "Unknown bot or status")
    store.set_bot_status(bid, status)
    return back(f"/bots/{bid}", "Bot " + ("started" if status == "running" else "paused"))


@app.post("/bots/{bid}/reset")
def bot_reset(bid: int, confirm: str = Form("")):
    if confirm.strip() != "RESET":
        return back(f"/bots/{bid}", "Type RESET to confirm")
    engine.reset_account(bid)
    return back(f"/bots/{bid}", "Paper account reset")


# ── Backtest Lab ─────────────────────────────────────────────────────────────

ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")


def pct(v, digits=1):
    return "–" if v is None else f"{v:+.{digits}f}%"


def fx(v, digits=2):
    return "–" if v is None else f"{v:,.{digits}f}"


def date_ms(ms):
    return datetime.fromtimestamp(ms / 1000, UK_TZ).strftime("%Y-%m-%d")


def datetime_ms(ms):
    return datetime.fromtimestamp(ms / 1000, UK_TZ).strftime("%Y-%m-%d %H:%M")


def iso_local(iso_str):
    """A stored UTC ISO-8601 string (futures trade times, event log, backtest run timestamps) ->
    'YYYY-MM-DD HH:MM' in UK local time. Replaces the naive `x[:16].replace('T',' ')` slicing
    those used before, which could only ever show UTC."""
    if not iso_str:
        return "–"
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(UK_TZ).strftime("%Y-%m-%d %H:%M")


templates.env.filters["pct"] = pct
templates.env.filters["fx"] = fx
templates.env.filters["date_ms"] = date_ms
templates.env.filters["datetime_ms"] = datetime_ms
templates.env.filters["iso_local"] = iso_local


_RUN_SELECT = ("SELECT r.id, r.created, r.kind, r.status, r.progress, r.message, r.config, r.brief, "
               "r.strategy_id, st.name AS strategy_name FROM bt_runs r "
               "LEFT JOIN strategies st ON st.id = r.strategy_id ")


def _run_rows(rows) -> list[dict]:
    """bt_runs rows (from _RUN_SELECT) shaped for the run tables."""
    out = []
    for r in rows:
        conf = json.loads(r["config"])
        out.append({"id": r["id"], "created": iso_local(r["created"]), "kind": r["kind"],
                    "status": r["status"], "progress": r["progress"], "message": r["message"],
                    "strategy": conf.get("strategy"), "strategy_id": r["strategy_id"],
                    "strategy_name": r["strategy_name"],
                    "symbol_list": conf.get("symbols", []),
                    "brief": json.loads(r["brief"]) if r["brief"] else {}})
    return out


def _running_job():
    with database.db() as c:
        return c.execute("SELECT id, kind FROM bt_runs WHERE status IN ('queued','running') "
                         "ORDER BY id LIMIT 1").fetchone()


def _spawn_worker(run_id: int) -> None:
    """Jobs run in their own process so a long backtest can't slow the site or the live bot."""
    os.makedirs(DATA_DIR, exist_ok=True)
    log = open(os.path.join(DATA_DIR, "backtest_worker.log"), "ab")
    subprocess.Popen([sys.executable, "-m", "app.backtest.worker", str(run_id)],
                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, close_fds=True)


def _date_to_ms(raw: str):
    try:
        return int(datetime.strptime(raw.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    except (ValueError, AttributeError):
        return None


@app.get("/backtest")
def backtest_page(request: Request, msg: str = "", strategy_id: int = 0):
    """`strategy_id` prefills the form from a saved strategy; otherwise the first one (or defaults)."""
    cov = btdata.coverage()
    saved = store.list_strategies()
    chosen = store.get_strategy(strategy_id) if strategy_id else (saved[0] if saved else None)
    if chosen:
        cfg = store.strategy_config(chosen)
    else:
        first = next(iter(STRATEGIES))
        cfg = {"strategy": first, "params": dict(STRATEGIES[first].default_params),
               "pairs": store.DEFAULT_PAIRS, "timeframe": store.DEFAULT_TIMEFRAME, **store.RISK_DEFAULTS}
    strat = STRATEGIES[cfg["strategy"]]
    grid = strat.grid_for({**strat.default_params, **cfg["params"]})
    with database.db() as c:
        rows = c.execute(_RUN_SELECT + "WHERE r.kind IN ('audit','oos') ORDER BY r.id DESC LIMIT 30").fetchall()
        last_data = c.execute("SELECT id, status, message, created FROM bt_runs WHERE kind='data' "
                              "ORDER BY id DESC LIMIT 1").fetchone()
    runs_list = _run_rows(rows)
    return render(request, "backtest.html", "backtest", msg=msg, cov=cov, cfg=cfg, strat=strat,
                  saved=saved, chosen_id=chosen["id"] if chosen else 0, link_id=chosen["id"] if chosen and strategy_id else 0, all_pairs=store.all_pairs(),
                  strategies=STRATEGIES, params=cfg["params"],
                  tfs=sorted({c["tf"] for c in cov}), runs=runs_list, last_data=last_data,
                  active_job=_running_job(), taker_fee_pct=cfg["taker_fee"] * 100,
                  grid=grid, grid_size=len(btruns.expand_grid(strat, cfg["params"], grid)) if grid else 0)


@app.post("/backtest/data")
def backtest_data(tf: str = Form("4h")):
    if _running_job():
        return back("/backtest", "Another job is still running. Wait for it to finish.")
    if tf not in ("1h", "4h", "1d"):
        return back("/backtest", "Unsupported timeframe")
    rid = btruns.create_run("data", {"symbols": store.all_pairs(), "timeframes": [tf]})
    _spawn_worker(rid)
    return RedirectResponse(f"/backtest/run/{rid}", status_code=303)


@app.post("/backtest/run")
async def backtest_run(request: Request):
    form = await request.form()
    if _running_job():
        return back("/backtest", "Another job is still running. Wait for it to finish.")
    kind = "oos" if form.get("kind") == "oos" else "audit"
    sname = form.get("strategy", "")
    if sname not in STRATEGIES:
        return back("/backtest", "Unknown strategy")
    strat = STRATEGIES[sname]
    tf = form.get("tf", "4h")
    symbols = [s.strip().upper() for s in re.split(r"[,\s]+", form.get("symbols", "")) if s.strip()]
    have = {(c["symbol"], c["tf"]) for c in btdata.coverage()}
    missing = [s for s in symbols if (s, tf) not in have]
    if not symbols or missing:
        return back("/backtest", "No stored " + tf + " data for: " + (", ".join(missing) or "any symbol")
                    + ". Use 'Download or update data' first.")
    cur = store.RISK_DEFAULTS
    cfg = {
        "capital": _num(form.get("capital"), 10, 10_000_000, cur["starting_capital"]),
        "risk_pct": _num(form.get("risk_pct"), 0.1, 10, cur["risk_pct"]),
        "leverage": _num(form.get("leverage"), 1, 20, cur["leverage"]),
        "max_margin_pct": _num(form.get("max_margin_pct"), 5, 100, cur["max_margin_pct"]),
        "max_simultaneous": _num(form.get("max_simultaneous"), 1, 10, cur["max_simultaneous"], int),
        "taker_fee": _num(form.get("taker_fee_pct"), 0, 1, cur["taker_fee"] * 100) / 100,
        "slippage_pct": _num(form.get("slippage_pct"), 0, 5, cur["slippage_pct"]),
        "max_hold_hours": _num(form.get("max_hold_hours"), 1, 720, cur["max_hold_hours"]),
        "max_funding_abs": cur["max_funding_abs"],
    }
    params = {}
    for key, default in strat.default_params.items():
        cast = int if isinstance(default, int) and not isinstance(default, bool) else float
        params[key] = _num(form.get("p_" + key), 0, 1_000_000, default, cast)
    config = {
        "strategy": sname, "symbols": symbols, "tf": tf, "cfg": cfg, "params": params,
        "n_random": _num(form.get("n_random"), 0, 500, 200, int),
        "start_ms": _date_to_ms(form.get("start", "")), "end_ms": _date_to_ms(form.get("end", "")),
        "split_pct": _num(form.get("split_pct"), 30, 80, 60),
        "min_trades": _num(form.get("min_trades"), 5, 500, 20, int),
    }
    linked = store.get_strategy(_num(form.get("strategy_id"), 0, 10**9, 0, int))
    if linked and linked["type"] != sname:                 # the type was changed by hand: no longer that strategy
        linked = None
    rid = btruns.create_run(kind, config, linked["id"] if linked else None)
    _spawn_worker(rid)
    return RedirectResponse(f"/backtest/run/{rid}", status_code=303)


def _add_funnel_bars(block):
    """Attach a log-scaled bar width to each funnel row (counts span several orders of magnitude)."""
    import math
    rows = block.get("funnel") if block else None
    if rows:
        top = math.log10(max(rows[0]["total"], 1) + 1)
        for r in rows:
            r["bar"] = round(100 * math.log10(r["total"] + 1) / top, 1) if top else 0
    return block


@app.get("/backtest/run/{rid}")
def backtest_run_page(request: Request, rid: int):
    with database.db() as c:
        row = c.execute("SELECT * FROM bt_runs WHERE id=?", (rid,)).fetchone()
    if row is None:
        return back("/backtest", "That run doesn't exist")
    config = json.loads(row["config"])
    res = json.loads(row["result"]) if row["status"] == "done" and row["result"] else None
    if res:
        for key in ("block", "train", "test", "untuned_test"):
            _add_funnel_bars(res.get(key))
    return render(request, "backtest_run.html", "backtest", run=dict(row), config=config, res=res,
                  strategies=STRATEGIES)


@app.get("/backtest/run/{rid}/status")
def backtest_run_status(rid: int):
    with database.db() as c:
        row = c.execute("SELECT status, progress, message FROM bt_runs WHERE id=?", (rid,)).fetchone()
    if row is None:
        return JSONResponse({"status": "missing"}, status_code=404)
    return {"status": row["status"], "progress": row["progress"], "message": row["message"]}


@app.get("/backtest/run/{rid}/trades.csv")
def backtest_trades_csv(rid: int, which: str = "test"):
    with database.db() as c:
        row = c.execute("SELECT result FROM bt_runs WHERE id=?", (rid,)).fetchone()
    if row is None or not row["result"]:
        return PlainTextResponse("no result", status_code=404)
    res = json.loads(row["result"])
    block = res.get("block") or res.get(which)
    if not block:
        return PlainTextResponse("no trades", status_code=404)
    buf = io.StringIO()
    cols = ["symbol", "entry_ts", "exit_ts", "entry", "exit", "outcome", "hold_h", "gross", "fees", "funding",
            "net", "r", "pct_move", "balance_after"]
    w = csv.writer(buf)
    w.writerow(["symbol", "entered_utc", "exited_utc", "entry", "exit", "outcome", "hours_held", "gross_pnl",
                "fees", "funding", "net_pnl", "r_multiple", "pct_move", "balance_after"])
    for t in block["trades"]:
        w.writerow([t[c] if c not in ("entry_ts", "exit_ts") else datetime_ms(t[c]) for c in cols])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="backtest_{rid}_trades.csv"'})


@app.post("/backtest/run/{rid}/delete")
def backtest_delete(rid: int):
    with database.db() as c:
        row = c.execute("SELECT status FROM bt_runs WHERE id=?", (rid,)).fetchone()
        if row and row["status"] in ("queued", "running"):
            return back(f"/backtest/run/{rid}", "That job is still running")
        c.execute("DELETE FROM bt_runs WHERE id=?", (rid,))
    return back("/backtest", f"Run #{rid} deleted")


# ── helpers ──────────────────────────────────────────────────────────────────

def _num(raw, lo, hi, default, cast=float):
    try:
        v = cast(raw)
    except (TypeError, ValueError):
        return default
    return min(max(v, lo), hi)


# ── Analysis ─────────────────────────────────────────────────────────────────
# Compare strategies side by side: the evidence pipeline, backtest vs out-of-sample vs live, and
# whether each live bot behaves the way its backtest said it would (the replay).

@app.get("/analysis")
def analysis_page(request: Request, s: list[int] = Query(default=[])):
    everything = store.list_strategies()
    chosen = [x for x in everything if x["id"] in s] if s else everything
    rows = [analysis.strategy_row(x) for x in chosen]
    bt_chart, oos_series, live_chart, live_rows = [], {}, [], []
    for r in rows:
        name = r["strategy"]["name"]
        if r["audit"]:
            ser = analysis.backtest_series(r["audit"]["id"])
            if ser:
                r["audit_series"] = ser
                bt_chart.append({"name": name, "t": ser["t"], "pct": ser["pct"]})
        if r["oos"]:
            r["oos_series"] = analysis.backtest_series(r["oos"]["id"])
        for b in store.bots_for_strategy(r["strategy"]["id"]):
            summ = analysis.live_summary(b)
            live_rows.append({**summ, "strategy": r["strategy"], "current": analysis.bot_matches(b["cfg"], r["strategy"])})
            live_chart.append({"name": f"{name} · bot #{b['id']}", **analysis.live_series(b)})
    return render(request, "analysis.html", "analysis", everything=everything, chosen_ids={x["id"] for x in chosen},
                  rows=rows, live_rows=live_rows, bt_chart=bt_chart, live_chart=live_chart,
                  min_live=analysis.MIN_LIVE_TRADES, stages=analysis.STAGES)


@app.get("/analysis/replay/{bid}")
def analysis_replay(request: Request, bid: int):
    bot = store.get_bot(bid)
    if bot is None:
        return back("/analysis", "That bot doesn't exist")
    try:
        rep, err = analysis.replay_bot(bot), ""
    except ValueError as e:
        rep, err = None, str(e)
    return render(request, "replay.html", "analysis", bot=bot, rep=rep, err=err,
                  strategy=store.get_strategy(bot["strategy_id"]))


# ── Strategies ───────────────────────────────────────────────────────────────
# A saved strategy = type + parameters + pairs + timeframe + risk/cost settings. Live bots and
# backtests are both run from one. Editing a strategy never changes a bot that is already running
# (it keeps the snapshot it started with); start a new bot to trade the new settings.

def _parse_strategy_form(form, existing: dict | None = None) -> tuple[dict | None, str]:
    """Validated strategy fields from a submitted form, or (None, error message)."""
    name = (form.get("name") or "").strip()[:80]
    stype = form.get("type", "")
    if not name:
        return None, "Give the strategy a name"
    if stype not in STRATEGIES:
        return None, "Unknown strategy type"
    base = existing["risk"] if existing else store.RISK_DEFAULTS
    strat = STRATEGIES[stype]
    old_params = existing["params"] if existing and existing["type"] == stype else strat.default_params
    params = {}
    for key, default in strat.default_params.items():
        if key in strat.flag_params:                       # a checkbox: present = on, absent = off
            params[key] = 1 if form.get("p_" + key) else 0
            continue
        cast = int if isinstance(default, int) and not isinstance(default, bool) else float
        params[key] = _num(form.get("p_" + key), 0, 1_000_000, old_params.get(key, default), cast)
    problem = strat.check(params)
    if problem:
        return None, problem
    symbols = [p.strip().upper() for p in re.split(r"[,\s]+", form.get("pairs", "")) if p.strip()]
    symbols = [x for x in dict.fromkeys(symbols) if re.fullmatch(r"[A-Z0-9]{1,20}USDT", x)]
    if not symbols:
        return None, "Choose at least one pair, like SOLUSDT"
    if len(symbols) > store.MAX_PAIRS:
        return None, f"Use at most {store.MAX_PAIRS} pairs per strategy"
    known = {r["symbol"] for r in market_symbols()}
    unknown = [x for x in symbols if known and x not in known]
    if unknown:
        return None, f"Bitget doesn't list a USDT perpetual for {', '.join(unknown)}"
    tf = form.get("timeframe", store.DEFAULT_TIMEFRAME)
    risk = {
        "starting_capital": _num(form.get("starting_capital"), 10, 1_000_000, base["starting_capital"]),
        "risk_pct": _num(form.get("risk_pct"), 0.1, 10, base["risk_pct"]),
        "max_simultaneous": _num(form.get("max_simultaneous"), 1, 10, base["max_simultaneous"], int),
        "leverage": _num(form.get("leverage"), 1, 20, base["leverage"], int),
        "max_margin_pct": _num(form.get("max_margin_pct"), 5, 100, base["max_margin_pct"]),
        "taker_fee": _num(form.get("taker_fee_pct"), 0, 1, base["taker_fee"] * 100) / 100,
        "slippage_pct": _num(form.get("slippage_pct"), 0, 5, base["slippage_pct"]),
        "max_hold_hours": _num(form.get("max_hold_hours"), 1, 720, base["max_hold_hours"], int),
        "max_funding_abs": _num(form.get("max_funding_pct"), 0, 5, base["max_funding_abs"] * 100) / 100,
        "daily_loss_limit_pct": _num(form.get("daily_loss_limit_pct"), 1, 100, base["daily_loss_limit_pct"]),
        "scan_minutes": _num(form.get("scan_minutes"), 1, 240, base["scan_minutes"], int),
    }
    return {"name": name, "type": stype, "params": params, "pairs": symbols,
            "timeframe": tf if tf in store.TIMEFRAMES else store.DEFAULT_TIMEFRAME, "risk": risk,
            "notes": (form.get("notes") or "").strip()[:1000]}, ""


def _strategy_form_ctx(s: dict | None) -> dict:
    """Template context for the create/edit form (s None = a blank new strategy)."""
    first = next(iter(STRATEGIES))
    s = s or {"id": None, "name": "", "type": first, "params": dict(STRATEGIES[first].default_params),
              "pairs": store.DEFAULT_PAIRS, "timeframe": store.DEFAULT_TIMEFRAME,
              "risk": dict(store.RISK_DEFAULTS), "notes": ""}
    plain_types = {k: t.plain(s["params"] if k == s["type"] else t.default_params) for k, t in STRATEGIES.items()}
    return {"s": s, "types": STRATEGIES, "plain_types": plain_types, "summary": STRATEGIES[s["type"]].describe(s["params"]),
            "max_pairs": store.MAX_PAIRS, "timeframes": store.TIMEFRAMES,
            "taker_fee_pct": s["risk"]["taker_fee"] * 100, "max_funding_pct": s["risk"]["max_funding_abs"] * 100}


@app.get("/strategies")
def strategies_page(request: Request, msg: str = "", archived: int = 0):
    rows = []
    for s in store.list_strategies(include_archived=bool(archived)):
        bots = store.bots_for_strategy(s["id"])
        with database.db() as c:
            last = c.execute("SELECT id, kind, brief FROM bt_runs WHERE strategy_id=? AND status='done' "
                             "AND kind IN ('audit','oos') ORDER BY id DESC LIMIT 1", (s["id"],)).fetchone()
        rows.append({**s, "label": STRATEGIES[s["type"]].label if s["type"] in STRATEGIES else s["type"],
                     "summary": STRATEGIES[s["type"]].describe(s["params"]),
                     "tagline": STRATEGIES[s["type"]].plain(s["params"])["tagline"],
                     "last_bt": {"id": last["id"], "kind": last["kind"], **json.loads(last["brief"] or "{}")} if last else None,
                     "n_bots": len(bots), "n_running": sum(b["status"] == "running" for b in bots)})
    return render(request, "strategies.html", "strategies", rows=rows, msg=msg, show_archived=bool(archived))


# NOTE: /strategies/new must be registered before /strategies/{sid}.
@app.get("/strategies/new")
def strategy_new(request: Request, msg: str = ""):
    return render(request, "strategy_form.html", "strategies", msg=msg, **_strategy_form_ctx(None))


@app.post("/strategies/new")
async def strategy_create(request: Request):
    fields, err = _parse_strategy_form(await request.form())
    if fields is None:
        return back("/strategies/new", err)
    sid = store.save_strategy(fields)
    return back(f"/strategies/{sid}", "Strategy created")


@app.get("/strategies/{sid}")
def strategy_page(request: Request, sid: int, msg: str = ""):
    s = store.get_strategy(sid)
    if s is None:
        return back("/strategies", "That strategy doesn't exist")
    bots = [bot_card(b) for b in store.bots_for_strategy(sid)]
    row = analysis.strategy_row(s)
    plain = explain.summary(s, row, analysis.backtest_series(row["audit"]["id"]) if row["audit"] else None)
    with database.db() as c:
        rows = c.execute(_RUN_SELECT + "WHERE r.strategy_id=? AND r.kind IN ('audit','oos') "
                                       "ORDER BY r.id DESC LIMIT 20", (sid,)).fetchall()
    return render(request, "strategy_form.html", "strategies", msg=msg, bots=bots, runs=_run_rows(rows), plain=plain, row=row,
                  missing_data=_missing_data(s), active_job=_running_job(), **_strategy_form_ctx(s))


@app.post("/strategies/{sid}")
async def strategy_update(sid: int, request: Request):
    existing = store.get_strategy(sid)
    if existing is None:
        return back("/strategies", "That strategy doesn't exist")
    fields, err = _parse_strategy_form(await request.form(), existing)
    if fields is None:
        return back(f"/strategies/{sid}", err)
    store.save_strategy(fields, sid)
    return back(f"/strategies/{sid}", "Saved. Running bots keep the settings they started with.")


@app.post("/strategies/{sid}/duplicate")
def strategy_duplicate(sid: int):
    s = store.get_strategy(sid)
    if s is None:
        return back("/strategies", "That strategy doesn't exist")
    new = store.save_strategy({**s, "name": (s["name"] + " (copy)")[:80]})
    return back(f"/strategies/{new}", "Duplicated. Change what you want to compare.")


@app.post("/strategies/{sid}/archive")
def strategy_archive(sid: int, archived: int = Form(1)):
    store.set_archived(sid, bool(archived))
    return back("/strategies", "Strategy archived" if archived else "Strategy restored")


def _missing_data(s: dict) -> list[str]:
    have = {(c["symbol"], c["tf"]) for c in btdata.coverage()}
    return [p for p in s["pairs"] if (p, s["timeframe"]) not in have]


@app.post("/strategies/{sid}/backtest")
def strategy_backtest(sid: int, kind: str = Form("audit")):
    s = store.get_strategy(sid)
    if s is None:
        return back("/strategies", "That strategy doesn't exist")
    if _running_job():
        return back(f"/strategies/{sid}", "Another job is still running. Wait for it to finish.")
    missing = _missing_data(s)
    if missing:
        return back(f"/strategies/{sid}", f"No stored {s['timeframe']} data for {', '.join(missing)}. "
                                          "Download it first.")
    rid = btruns.create_run("oos" if kind == "oos" else "audit", analysis.backtest_config(s), sid)
    _spawn_worker(rid)
    return RedirectResponse(f"/backtest/run/{rid}", status_code=303)


@app.post("/strategies/{sid}/data")
def strategy_download_data(sid: int):
    s = store.get_strategy(sid)
    if s is None:
        return back("/strategies", "That strategy doesn't exist")
    if _running_job():
        return back(f"/strategies/{sid}", "Another job is still running. Wait for it to finish.")
    rid = btruns.create_run("data", {"symbols": s["pairs"], "timeframes": [s["timeframe"]]})
    _spawn_worker(rid)
    return RedirectResponse(f"/backtest/run/{rid}", status_code=303)


@app.post("/strategies/{sid}/bots")
def strategy_start_bot(sid: int):
    if store.get_strategy(sid) is None:
        return back("/strategies", "That strategy doesn't exist")
    bid = store.create_bot(sid)
    return back(f"/bots/{bid}", "Paper bot started with its own account")
