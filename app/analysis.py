"""Comparing strategies: how far each has got through the evidence pipeline, and whether the
live paper bots behave the way their backtests said they would.

The pipeline (each stage needs the one before it, and only counts results obtained with the
strategy's CURRENT settings; a backtest of older settings says nothing about today's):

  1 Backtested            an audit exists
  2 Passes honesty checks the audit's verdict is not "no edge" and it made money after costs
  3 Held out-of-sample    the out-of-sample test, measured on the unseen period, also passes
  4 Proven live so far    a paper bot with these settings has MIN_LIVE_TRADES closed trades and is profitable
"""
import json
import math
from datetime import datetime, timezone

from app import strategy_store as store
from app.backtest import runs
from app.backtest.engine import simulate
from app.db import db
from app.futures import engine
from app.strategies import STRATEGIES

MIN_LIVE_TRADES = 30
STAGES = ["Not backtested", "Backtested", "Passes honesty checks", "Held out-of-sample", "Proven live so far"]
_RUN_KEYS = ("capital", "risk_pct", "leverage", "max_margin_pct", "max_simultaneous", "taker_fee",
             "slippage_pct", "max_hold_hours", "max_funding_abs")


# ── does a result belong to a strategy's current settings? ───────────────────

def backtest_config(s: dict) -> dict:
    """A backtest config that runs a saved strategy exactly as saved (its own costs and risk)."""
    r = s["risk"]
    return {
        "strategy": s["type"], "symbols": s["pairs"], "tf": s["timeframe"], "params": s["params"],
        "cfg": {"capital": r["starting_capital"], "risk_pct": r["risk_pct"], "leverage": r["leverage"],
                "max_margin_pct": r["max_margin_pct"], "max_simultaneous": r["max_simultaneous"],
                "taker_fee": r["taker_fee"], "slippage_pct": r["slippage_pct"],
                "max_hold_hours": r["max_hold_hours"], "max_funding_abs": r["max_funding_abs"]},
        "n_random": 200, "start_ms": None, "end_ms": None, "split_pct": 60, "min_trades": 20,
    }


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _params(strategy_type: str, params: dict) -> dict:
    return {**STRATEGIES[strategy_type].default_params, **params} if strategy_type in STRATEGIES else params


def run_matches(run_config: dict, s: dict) -> bool:
    """True if a stored backtest config is the strategy's current settings (type, params, pairs,
    timeframe, costs and risk). Window/baseline options are ignored."""
    ref = backtest_config(s)
    if run_config.get("strategy") != ref["strategy"] or run_config.get("tf") != ref["tf"]:
        return False
    if sorted(run_config.get("symbols", [])) != sorted(ref["symbols"]):
        return False
    a, b = _params(ref["strategy"], run_config.get("params", {})), _params(ref["strategy"], ref["params"])
    if set(a) != set(b) or not all(_same(a[k], b[k]) for k in a):
        return False
    return all(_same(run_config.get("cfg", {}).get(k), ref["cfg"][k]) for k in _RUN_KEYS)


def bot_matches(bot_cfg: dict, s: dict) -> bool:
    """True if a bot's settings snapshot is the strategy's current settings (scan interval ignored)."""
    ref = store.strategy_config(s)
    if bot_cfg.get("strategy") != ref["strategy"] or bot_cfg.get("timeframe") != ref["timeframe"]:
        return False
    if sorted(bot_cfg.get("pairs", [])) != sorted(ref["pairs"]):
        return False
    a, b = _params(ref["strategy"], bot_cfg.get("params", {})), _params(ref["strategy"], ref["params"])
    if set(a) != set(b) or not all(_same(a[k], b[k]) for k in a):
        return False
    return all(_same(bot_cfg.get(k), ref[k]) for k in store.RISK_DEFAULTS if k != "scan_minutes")


# ── evidence pipeline per strategy ───────────────────────────────────────────

def _passes(brief: dict | None) -> bool:
    return bool(brief) and brief.get("level") not in (None, "none") and (brief.get("return_pct") or 0) > 0


def stage_of(audit: dict | None, oos: dict | None, live: dict | None) -> int:
    """Highest pipeline stage reached (0-4). Inputs are the briefs of the current-settings audit,
    the current-settings out-of-sample run, and the live summary of the best matching bot."""
    if not audit:
        return 0
    if not _passes(audit):
        return 1
    if not _passes(oos):
        return 2
    if not (live and live["trades"] >= MIN_LIVE_TRADES and live["return_pct"] > 0):
        return 3
    return 4


def next_step(stage: int, audit, oos, live, stale: bool, has_bot: bool) -> tuple[str, str]:
    """(plain-language next step, tone good|warn|bad|'')."""
    if audit is None:
        return ("Settings changed since the last backtest: run a new audit" if stale else "Run an audit"), "warn"
    if stage == 1:
        return "No edge shown in the audit: revise the rules or drop it", "bad"
    if stage == 2:
        if oos is None:
            return "Run the out-of-sample test", "warn"
        return "The out-of-sample test showed no reliable edge: the audit result may be fitted to the past", "bad"
    if stage == 3:
        if not has_bot:
            return "Start a paper bot to forward-test it", "warn"
        n = live["trades"] if live else 0
        if live and n >= MIN_LIVE_TRADES:
            return "Live results disagree with the backtest: look at the replay", "bad"
        return f"Forward-testing: {n}/{MIN_LIVE_TRADES} live trades so far", "warn"
    return "Backtest, out-of-sample and live all agree so far. Keep it running", "good"


def _latest_runs(s: dict) -> dict:
    """Newest done audit and oos run for this strategy, split by whether they match today's settings."""
    with db() as c:
        rows = c.execute("SELECT id, kind, brief, config, created FROM bt_runs WHERE strategy_id=? "
                         "AND status='done' AND kind IN ('audit','oos') ORDER BY id DESC", (s["id"],)).fetchall()
    out = {"audit": None, "oos": None, "stale": False}
    for r in rows:
        if run_matches(json.loads(r["config"]), s):
            if out[r["kind"]] is None:
                out[r["kind"]] = {"id": r["id"], "created": r["created"], **json.loads(r["brief"] or "{}")}
        else:
            out["stale"] = True
    return out


def live_summary(bot: dict) -> dict:
    summ = engine.summary(bot["id"], bot["cfg"])
    started = datetime.fromisoformat(bot["started_at"])
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return {**summ, "bot_id": bot["id"], "name": bot["name"], "status": bot["status"],
            "halted": engine.get_state(bot["id"], "halted"),
            "days": (datetime.now(timezone.utc) - started).total_seconds() / 86400}


def strategy_row(s: dict) -> dict:
    """Everything the dashboard and analysis tables show about one strategy."""
    runs_ = _latest_runs(s)
    bots = store.bots_for_strategy(s["id"])
    current = [b for b in bots if bot_matches(b["cfg"], s)]
    lives = [live_summary(b) for b in current]
    live = max(lives, key=lambda x: (x["trades"], x["bot_id"])) if lives else None
    stage = stage_of(runs_["audit"], runs_["oos"], live)
    step, tone = next_step(stage, runs_["audit"], runs_["oos"], live, runs_["stale"], bool(current))
    return {"strategy": s, "label": STRATEGIES[s["type"]].label if s["type"] in STRATEGIES else s["type"],
            "tagline": STRATEGIES[s["type"]].plain(s["params"])["tagline"] if s["type"] in STRATEGIES else "",
            "audit": runs_["audit"], "oos": runs_["oos"], "stale": runs_["stale"], "live": live,
            "other_bots": len(bots) - len(current), "stage": stage, "stage_name": STAGES[stage],
            "next": step, "tone": tone}


# ── chart series ─────────────────────────────────────────────────────────────

def backtest_series(run_id: int) -> dict | None:
    """{'t': [ms], 'pct': [% return]} for an audit run's equity curve, plus its honesty-check counts."""
    with db() as c:
        row = c.execute("SELECT result FROM bt_runs WHERE id=?", (run_id,)).fetchone()
    if not row or not row["result"]:
        return None
    res = json.loads(row["result"])
    block = res.get("block") or res.get("test")
    if not block:
        return None
    cap = res["cfg"]["capital"]
    eq = block["equity"]
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for ck in res.get("checks", []):
        counts[ck["status"]] = counts.get(ck["status"], 0) + 1
    return {"t": eq["t"], "pct": [round((v / cap - 1) * 100, 3) for v in eq["v"]], "checks": counts,
            "by_symbol": block.get("by_symbol", {}), "bh": block["bh"]["summary"], "summary": block["summary"],
            "beats_random": (block.get("random") or {}).get("pctile_return"),
            "reasons": res.get("verdict", {}).get("reasons", [])}


def live_series(bot: dict) -> dict:
    """{'t': [ms], 'pct': [...]} of a bot's realised equity: start, then each closed trade."""
    start_bal = float(bot["cfg"]["starting_capital"])
    started = _iso_ms(bot["started_at"])
    with db() as c:
        rows = c.execute("SELECT exit_time, balance_after FROM futures_trades WHERE bot_id=? AND status='closed' "
                         "ORDER BY exit_time", (bot["id"],)).fetchall()
    t, pct = [started], [0.0]
    for r in rows:
        t.append(_iso_ms(r["exit_time"]))
        pct.append(round((r["balance_after"] / start_bal - 1) * 100, 3))
    now_bal = float(engine.get_state(bot["id"], "balance", start_bal))
    t.append(int(datetime.now(timezone.utc).timestamp() * 1000))
    pct.append(round((now_bal / start_bal - 1) * 100, 3))
    return {"t": t, "pct": pct}


def _iso_ms(iso_str: str) -> int:
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ── replay: what would the backtest have done over the bot's live window? ─────

def match_trades(live: list[dict], bt: list[dict], tf_ms: int) -> dict:
    """Pair each live trade with the backtest trade from the same signal. A backtest enters at the
    open of the candle AFTER the signal candle, so the key is (symbol, signal candle open + timeframe).
    Live rows need `symbol` and `signal_ms`; backtest rows `symbol` and `entry_ts`."""
    by_key = {(t["symbol"], t["entry_ts"]): t for t in bt}
    used, matched, live_only = set(), [], []
    for lt in live:
        key = (lt["symbol"], lt["signal_ms"] + tf_ms) if lt.get("signal_ms") is not None else None
        if key in by_key and key not in used:
            used.add(key)
            matched.append((lt, by_key[key]))
        else:
            live_only.append(lt)
    bt_only = [t for t in bt if (t["symbol"], t["entry_ts"]) not in used]
    return {"matched": matched, "live_only": live_only, "bt_only": bt_only}


def _live_trades(bot: dict) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT * FROM futures_trades WHERE bot_id=? ORDER BY entry_time", (bot["id"],)).fetchall()
    out = []
    for t in rows:
        risk = t["units"] * (t["entry_price"] - t["stop"])
        out.append({**dict(t), "signal_ms": _iso_ms(t["signal_time"]) if t["signal_time"] else None,
                    "r": (t["net_pnl"] / risk) if t["net_pnl"] is not None and risk > 0 else None,
                    "mark": engine.get_state(bot["id"], f"mark:{t['symbol']}")})
    return out


def window_start_ms(started_ms: int, live: list[dict]) -> int:
    """Where the replay window begins: when the bot started, or earlier if it holds a trade whose
    signal came before that (a bot migrated from the original single bot 'starts' at its first
    trade, but that trade's backtest twin entered a little earlier, on the candle open)."""
    signals = [t["signal_ms"] for t in live if t.get("signal_ms") is not None]
    return min([started_ms] + signals)


def replay_bot(bot: dict) -> dict:
    """Run the backtest simulator over exactly the period this bot has been live, with its own
    settings, and compare it trade by trade with what the bot really did. Raises ValueError
    with a plain message when there isn't enough to compare yet."""
    cfg = bot["cfg"]
    strat = STRATEGIES[cfg["strategy"]]
    market = runs.load_market(cfg["pairs"], cfg["timeframe"])
    bt_cfg = runs.bt_config({"capital": cfg["starting_capital"], **{k: cfg[k] for k in _RUN_KEYS if k in cfg and k != "capital"}})
    live = _live_trades(bot)
    start_ms = window_start_ms(_iso_ms(bot["started_at"]), live)
    end_ms = int(market.T[-1]) + market.tf_ms
    if end_ms - start_ms < market.tf_ms:
        raise ValueError("This bot started less than one candle ago; nothing to compare yet.")
    params = {**strat.default_params, **cfg["params"]}
    sim = simulate(market, market.signals(strat, params), bt_cfg, start_ms, end_ms)

    m = match_trades(live, sim.trades, market.tf_ms)
    start_bal = float(cfg["starting_capital"])
    balance = float(engine.get_state(bot["id"], "balance", start_bal))
    unreal = sum((float(t["mark"]) - t["entry_price"]) * t["units"] for t in live if t["status"] == "open" and t["mark"])
    live_ret = (balance + unreal - start_bal) / start_bal * 100

    slip = [(lt["entry_price"] / bt["entry"] - 1) * 100 for lt, bt in m["matched"]]
    r_gap = [lt["r"] - bt["r"] for lt, bt in m["matched"] if lt["r"] is not None]
    n_live, n_bt = len(live), len(sim.trades)
    return {
        "window": {"start": start_ms, "end": end_ms, "days": (end_ms - start_ms) / 86_400_000},
        "live_return_pct": live_ret, "bt_return_pct": (sim.final_balance / bt_cfg.capital - 1) * 100,
        "n_live": n_live, "n_bt": n_bt, "n_matched": len(m["matched"]),
        "match_share": len(m["matched"]) / max(n_live, n_bt) if max(n_live, n_bt) else None,
        "avg_entry_slip_pct": sum(slip) / len(slip) if slip else None,
        "avg_r_gap": sum(r_gap) / len(r_gap) if r_gap else None,
        "matched": m["matched"], "live_only": m["live_only"], "bt_only": m["bt_only"],
        "tf_ms": market.tf_ms, "data_last_ms": int(market.T[-1]),
    }
