"""Backtest jobs: data download, audit, and out-of-sample test.

These run in a separate process (see worker.py) so a long job can't slow the website
or the live paper bot. Results are stored as JSON in the bt_runs table.

Two ways to use history honestly:
  AUDIT  run the strategy exactly as configured over the whole history, with no tuning,
         and compare it with buy-and-hold and with random entries.
  OOS    pick parameters using ONLY an early "training" period, then measure them once
         on a later period the tuning never saw. The out-of-sample number is the one
         to believe; the training number is flattering by construction.
"""
import dataclasses
import itertools
import json
import math
import multiprocessing as mp
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.backtest import data, metrics
from app.backtest.engine import BtConfig, Market, simulate
from app.db import db
from app.strategies import STRATEGIES, get_strategy

MAX_TRADES_STORED = 3000
WARMUP_BARS = 60


# ── small helpers ────────────────────────────────────────────────────────────

def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_json(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"not JSON serialisable: {type(o)}")
    return json.dumps(obj, default=default, allow_nan=False)


def create_run(kind: str, config: dict, strategy_id: int | None = None) -> int:
    with db() as c:
        cur = c.execute("INSERT INTO bt_runs(created, kind, status, progress, message, config, strategy_id) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (_iso(), kind, "queued", 0.0, "Queued", json.dumps(config), strategy_id))
        return cur.lastrowid


class Progress:
    """Callable that writes progress into the run's row (throttled)."""

    def __init__(self, run_id: int):
        self.run_id, self._last, self.frac, self.msg = run_id, 0.0, 0.0, ""

    def __call__(self, frac: float | None = None, msg: str | None = None, force: bool = False):
        if frac is not None:
            self.frac = max(0.0, min(float(frac), 1.0))
        if msg is not None:
            self.msg = msg
        now = time.time()
        if force or now - self._last >= 0.7:
            self._last = now
            with db() as c:
                c.execute("UPDATE bt_runs SET progress=?, message=?, status='running' WHERE id=?",
                          (self.frac, self.msg, self.run_id))

    def sub(self, lo: float, hi: float):
        """A callback that maps 0..1 onto the lo..hi slice of the whole job."""
        return lambda f, msg=None: self(lo + (hi - lo) * f, msg)


def bt_config(d: dict) -> BtConfig:
    known = {f.name for f in dataclasses.fields(BtConfig)}
    return BtConfig(**{k: v for k, v in d.items() if k in known})


def load_market(symbols: list[str], tf: str) -> Market:
    frames, fund = {}, {}
    for s in symbols:
        df = data.load_candles(s, tf)
        if df.empty:
            raise ValueError(f"No {tf} candles stored for {s}. Download the data first.")
        frames[s] = df
        fund[s] = data.load_funding(s)
    return Market(frames, fund, data.TF_MS[tf])


def default_window(market: Market, warmup_bars: int = WARMUP_BARS) -> tuple[int, int]:
    """From when EVERY symbol has data (plus the strategy's indicator warm-up) to the latest closed candle."""
    first = max(int(df["ts_ms"].iloc[0]) for df in market.frames.values())
    return first + warmup_bars * market.tf_ms, int(market.T[-1]) + market.tf_ms


# ── one evaluated block: strategy vs buy-and-hold vs random ──────────────────

def funnel(strategy, market: Market, params: dict, start_ms: int, end_ms: int, trades_taken: int):
    """How many candles survive each part of the entry rule (summed over symbols)."""
    tf = market.tf_ms
    rows, per_sym = None, {}
    for sym, df in market.frames.items():
        conds = strategy.explain(df, params)
        if conds is None:
            return None
        ts = df["ts_ms"].to_numpy(dtype=np.int64)
        in_win = (ts >= start_ms - tf) & (ts < end_ms - tf)     # signal candle whose next open is in the window
        acc = in_win.copy()
        counts = [("Candles examined", int(acc.sum()))]
        for label, arr in conds:
            acc = acc & np.asarray(arr, dtype=bool)
            counts.append((label, int(acc.sum())))
        per_sym[sym] = counts
        if rows is None:
            rows = [{"label": lab, "total": 0, "by_symbol": {}} for lab, _ in counts]
        for r, (_, n) in zip(rows, counts):
            r["total"] += n
            r["by_symbol"][sym] = n
    rows.append({"label": "Trades actually opened (after slot, margin and funding limits)",
                 "total": trades_taken, "by_symbol": {}})
    return rows


def stress_table(market: Market, sigs: dict, cfg: BtConfig, start: int, end: int) -> list:
    variants = [
        ("Your costs (baseline)", {}),
        ("No slippage (optimistic)", {"slippage_pct": 0.0}),
        ("2x slippage", {"slippage_pct": cfg.slippage_pct * 2}),
        ("4x slippage", {"slippage_pct": cfg.slippage_pct * 4}),
        ("1.5x trading fees", {"taker_fee": cfg.taker_fee * 1.5}),
        ("Worst-case funding (always pay)", {"funding_worst_case": True}),
        ("Enter one candle later", {"entry_delay_bars": 1}),
    ]
    rows = []
    for label, over in variants:
        c = dataclasses.replace(cfg, **over)
        s = metrics.summarize(simulate(market, sigs, c, start, end), c, market.tf_ms)
        rows.append({"label": label, "return_pct": s["return_pct"], "trades": s["trades"],
                     "profit_factor": s["profit_factor"], "mean_r": s["mean_r"], "max_dd_pct": s["max_dd_pct"]})
    return rows


def evaluate(market: Market, strategy, params: dict, cfg: BtConfig, start: int, end: int,
             n_random: int, progress=None, seed: int = 7, stress: bool = True) -> dict:
    tf = market.tf_ms
    sigs = market.signals(strategy, params)
    sim = simulate(market, sigs, cfg, start, end)
    summ = metrics.summarize(sim, cfg, tf)
    bh = metrics.buy_and_hold(market, cfg, start, end)
    rnd = metrics.random_baseline(market, sigs, cfg, start, end, summ, sim, n_sims=n_random, seed=seed,
                                  progress=progress)
    # Second baseline, only for strategies with a regime filter: random entries INSIDE that regime.
    has_regime = any((~sigs[s]["C"] & market.V[s]).any() for s in market.symbols)
    rnd_regime = (metrics.random_baseline(market, sigs, cfg, start, end, summ, sim, n_sims=n_random,
                                          seed=seed + 1000, regime=True)
                  if has_regime else None)
    boot = metrics.bootstrap_mean_r([t["r"] for t in sim.trades])
    trades = [{
        "symbol": t["symbol"], "entry_ts": t["entry_ts"], "exit_ts": t["exit_ts"], "entry": t["entry"],
        "exit": t["exit"], "outcome": t["outcome"], "hold_h": t["hold_h"], "gross": t["gross"],
        "fees": t["fee_open"] + t["fee_close"], "funding": t["funding"], "net": t["net"], "r": t["r"],
        "pct_move": t["pct_move"], "balance_after": t["balance_after"],
    } for t in sim.trades]
    return {
        "window": {"start": start, "end": end, "bars": sim.bars, "years": summ["years"]},
        "params": params, "summary": summ,
        "equity": metrics.downsample(sim.eq_ts, sim.eq),
        "bh": {"summary": bh["summary"], "per_symbol": bh["per_symbol"],
               "equity": metrics.downsample(bh["eq_ts"], bh["eq"])},
        "random": rnd, "random_regime": rnd_regime, "bootstrap": boot,
        "by_symbol": metrics.by_symbol(sim.trades),
        "by_quarter": metrics.by_quarter(sim, bh, cfg.capital),
        "funnel": funnel(strategy, market, params, start, end, len(sim.trades)),
        "stress": stress_table(market, sigs, cfg, start, end) if stress and sim.trades else [],
        "trades": trades[-MAX_TRADES_STORED:], "trades_truncated": len(trades) > MAX_TRADES_STORED,
    }


# ── honesty checks and the verdict ───────────────────────────────────────────

def _chk(name, status, detail, critical=False):
    return {"name": name, "status": status, "detail": detail, "critical": critical}


def make_checks(block: dict, oos: dict | None = None) -> list:
    s = block["summary"]
    n = s["trades"]
    checks = []
    if n == 0:
        return [_chk("The strategy traded", "fail",
                     "It opened zero trades in this period, so there is nothing to evaluate. "
                     "See the signal funnel to find which condition blocks entries.", True)]

    # 1. enough trades to say anything
    checks.append(_chk("Enough trades", "pass" if n >= 100 else "warn" if n >= 30 else "fail",
                       f"{n} trades. Fewer than 30 is too few to tell skill from luck; 100+ starts to be meaningful.",
                       True))
    # 2. statistical evidence
    b = block["bootstrap"]
    if b is None:
        checks.append(_chk("Statistical evidence", "fail", "Too few trades (under 8) to estimate anything.", True))
    else:
        ok = b["lo"] > 0
        checks.append(_chk("Statistical evidence", "pass" if ok else "fail",
                           f"Average result per trade is {b['mean']:+.2f}R, 95% range {b['lo']:+.2f}R to {b['hi']:+.2f}R. "
                           + ("The whole range is above zero." if ok else
                              "The range includes zero, so luck can't be ruled out."), True))
    # 3. beats random entries with the same exits
    r = block["random"]
    if r is None:
        checks.append(_chk("Beats random entries", "warn", "Not measured (too few trades)."))
    else:
        p = r["pctile_return"]
        checks.append(_chk("Beats random entries", "pass" if p >= 95 else "warn" if p >= 80 else "fail",
                           f"Better than {p:.0f}% of {r['n']} runs that entered at random with identical exits and costs "
                           f"(random median {r['median']:+.1f}%). Below ~95% means the entry signal adds no clear edge.",
                           True))
    # 3b. does the TRIGGER add value beyond the regime filter? (diagnostic, not a key check)
    rr = block.get("random_regime")
    if rr is not None:
        p = rr["pctile_return"]
        checks.append(_chk("Trigger adds value beyond the regime filter", "pass" if p >= 90 else "warn" if p >= 70 else "fail",
                           f"Better than {p:.0f}% of {rr['n']} runs that entered at random but ONLY while the strategy's own "
                           f"regime filter was on (random median {rr['median']:+.1f}%). If this is low, the result comes from "
                           f"being in that regime, not from the entry rule."))
    # 4. vs buy and hold
    bh = block["bh"]["summary"]["return_pct"]
    ret = s["return_pct"]
    if ret > bh and ret > 0:
        st, txt = "pass", f"Made {ret:+.1f}% versus {bh:+.1f}% for buy-and-hold."
    elif ret > bh:
        st, txt = "warn", (f"Lost {ret:+.1f}%, but buy-and-hold lost more ({bh:+.1f}%). "
                           f"Doing less in a falling market is not the same as having an edge.")
    else:
        st, txt = "fail", f"Made {ret:+.1f}% versus {bh:+.1f}% for simply holding the same coins."
    checks.append(_chk("Beats buy-and-hold", st, txt))
    # 5. costs
    stress = {x["label"]: x for x in block["stress"]}
    s2 = stress.get("2x slippage")
    if ret <= 0:
        checks.append(_chk("Survives higher costs", "fail", "Already loses money at your configured costs.", True))
    elif s2 is not None:
        ok = s2["return_pct"] > 0
        checks.append(_chk("Survives higher costs", "pass" if ok else "fail",
                           f"At 2x slippage the result is {s2['return_pct']:+.1f}%. "
                           + ("Still positive." if ok else "The profit disappears, so it depends on very cheap fills."), True))
    # 6. late entry
    late = stress.get("Enter one candle later")
    if late is not None and ret > 0:
        ok = late["return_pct"] > 0
        checks.append(_chk("Not fragile to timing", "pass" if ok else "warn",
                           f"Entering one candle later gives {late['return_pct']:+.1f}%."
                           + ("" if ok else " The result depends on catching the exact candle.")))
    # 7. concentration
    if s["top3_share"] is not None and ret > 0:
        ok = s["top3_share"] < 0.5
        checks.append(_chk("Not carried by a few trades", "pass" if ok else "warn",
                           f"The best 3 trades make up {s['top3_share'] * 100:.0f}% of all profit."
                           + ("" if ok else " Remove them and the result may vanish.")))
    # 8. too good to be true
    flags = []
    if s["profit_factor"] and s["profit_factor"] > 3 and n < 200:
        flags.append(f"profit factor {s['profit_factor']:.1f}")
    if s["sharpe"] and s["sharpe"] > 3:
        flags.append(f"Sharpe {s['sharpe']:.1f}")
    if s["win_rate"] and s["win_rate"] > 80:
        flags.append(f"win rate {s['win_rate']:.0f}%")
    if ret > 300:
        flags.append(f"return {ret:.0f}%")
    if flags:
        checks.append(_chk("Too good to trust", "warn",
                           "Unusually strong numbers (" + ", ".join(flags) + "). Real strategies rarely look like this; "
                           "suspect a bug or overfitting before celebrating."))

    # out-of-sample specific
    if oos is not None:
        tr, te = oos["train_mean_r"], oos["test_mean_r"]
        if te is None:
            checks.append(_chk("Holds up out-of-sample", "fail", "No trades in the unseen test period.", True))
        elif te <= 0:
            checks.append(_chk("Holds up out-of-sample", "fail",
                               f"Average result per trade fell from {tr:+.2f}R in training to {te:+.2f}R on unseen data.", True))
        elif tr is not None and te < 0.5 * tr:
            checks.append(_chk("Holds up out-of-sample", "warn",
                               f"Still positive ({te:+.2f}R) but less than half of the training figure ({tr:+.2f}R): "
                               f"typical when tuning captured some noise.", True))
        else:
            checks.append(_chk("Holds up out-of-sample", "pass",
                               f"{te:+.2f}R per trade on unseen data versus {tr:+.2f}R in training.", True))
        luck, best_t = oos.get("luck_t"), oos.get("train_t")
        if luck and best_t is not None:
            ok = best_t > luck
            checks.append(_chk("Training result beats what luck alone produces", "pass" if ok else "warn",
                               f"Best of {oos['n_tried']} tries scored t={best_t:.1f} in training. Picking the best of that "
                               f"many random-ish candidates gives about t={luck:.1f} by chance alone."
                               + ("" if ok else " So even the training result is not impressive for the number of tries.")))
        rho = oos.get("spearman")
        if rho is not None:
            st = "pass" if rho > 0.3 else "warn" if rho > 0 else "fail"
            checks.append(_chk("Training ranking predicted the test ranking", st,
                               f"Rank correlation between how parameter sets did in training and on unseen data is {rho:+.2f}. "
                               f"Near zero means picking the training winner told you nothing about the future."))
    return checks


def make_verdict(checks: list, n_trades: int) -> dict:
    if n_trades == 0:
        return {"level": "none", "headline": "No trades, nothing to evaluate",
                "reasons": ["The rules never produced a trade in this period."]}
    critical_fail = [c for c in checks if c["critical"] and c["status"] == "fail"]
    critical_warn = [c for c in checks if c["critical"] and c["status"] == "warn"]
    if critical_fail:
        level, head = "none", "No reliable edge shown"
    elif critical_warn:
        level, head = "weak", "Weak evidence, not enough to trust"
    else:
        level, head = "moderate", "Some evidence, worth forward-testing before anything else"
    return {"level": level, "headline": head,
            "reasons": [c["name"] + ": " + c["detail"] for c in (critical_fail + critical_warn)][:4]}


# ── parameter tuning on the training period (parallel grid search) ───────────

_G: dict = {}          # filled in the parent before forking, read by the workers


def _grid_eval(combo: dict) -> dict:
    m, strat, cfg, windows = _G["market"], _G["strat"], _G["cfg"], _G["windows"]
    sigs = m.signals(strat, combo)
    out = {"params": combo}
    for name, (a, b) in windows.items():
        s = metrics.summarize(simulate(m, sigs, cfg, a, b), cfg, m.tf_ms)
        out[name] = {k: s[k] for k in ("trades", "return_pct", "max_dd_pct", "profit_factor", "mean_r", "t_stat", "win_rate")}
    return out


def expand_grid(strategy, base_params: dict, grid: dict) -> list:
    keys = list(grid)
    combos = [{**base_params, **dict(zip(keys, vals))} for vals in itertools.product(*(grid[k] for k in keys))]
    return [c for c in combos if strategy.is_valid(c)]


def run_grid(market, strategy, combos, cfg, windows: dict, progress, workers: int = 3) -> list:
    _G.update(market=market, strat=strategy, cfg=cfg, windows=windows)
    results = []
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_grid_eval, combos, chunksize=6)):
            results.append(res)
            if progress and (i % 8 == 0 or i == len(combos) - 1):
                progress((i + 1) / len(combos), f"Tuning on the training period: {i + 1} of {len(combos)} parameter sets")
    return results


def select_and_diagnose(results: list, min_trades: int) -> dict:
    ok = [r for r in results if r["train"]["trades"] >= min_trades and r["train"]["t_stat"] is not None]
    selected = max(ok, key=lambda r: (r["train"]["t_stat"], r["train"]["trades"])) if ok else None

    both = [(r["train"]["t_stat"], r["test"]["t_stat"]) for r in results
            if r["train"]["t_stat"] is not None and r["test"]["t_stat"] is not None]
    rho = None
    if len(both) >= 10:
        a, b = zip(*both)
        rho = pd.Series(a).rank().corr(pd.Series(b).rank())     # Spearman = Pearson on ranks (no scipy needed)
        rho = None if pd.isna(rho) else float(rho)

    test_ts = [r["test"]["t_stat"] for r in results if r["test"]["t_stat"] is not None]
    rank_pct = None
    if selected is not None and selected["test"]["t_stat"] is not None and test_ts:
        rank_pct = float(np.mean([t < selected["test"]["t_stat"] for t in test_ts]) * 100)
    test_returns = [r["test"]["return_pct"] for r in results if r["test"]["trades"] > 0]

    top = sorted(ok, key=lambda r: -r["train"]["t_stat"])[:12]
    return {
        "n_tried": len(results), "n_eligible": len(ok), "min_trades": min_trades,
        "selected": selected, "spearman": rho, "selected_test_rank_pct": rank_pct,
        "expected_best_t_by_luck": math.sqrt(2 * math.log(len(ok))) if len(ok) > 1 else None,
        "median_test_return": float(np.median(test_returns)) if test_returns else None,
        "share_profitable_test": float(np.mean([x > 0 for x in test_returns]) * 100) if test_returns else None,
        "top": top,
    }


# ── the three job types ──────────────────────────────────────────────────────

def _data_info(market: Market) -> dict:
    cov = {(c["symbol"], c["tf"]): c for c in data.coverage()}
    info = {}
    for sym in market.symbols:
        c = cov.get((sym, "4h")) or next((v for (s, _), v in cov.items() if s == sym), None)
        info[sym] = {"first": c["first"] if c else "", "last": c["last"] if c else "",
                     "candles": int(len(market.frames[sym])), "missing": c["missing"] if c else 0,
                     "funding_real": c["funding_real"] if c else 0, "funding_rows": c["funding_rows"] if c else 0}
    return info


def run_data(config: dict, progress: Progress) -> dict:
    symbols, tfs = config["symbols"], config.get("timeframes", ["4h"])
    steps = len(symbols) * (len(tfs) + 2)
    done, added = 0, {}
    for sym in symbols:
        for tf in tfs:
            n = data.update_candles(sym, tf, progress=lambda m: progress(done / steps, m))
            added[f"{sym} {tf} candles"] = n
            done += 1
        added[f"{sym} funding (Binance proxy)"] = data.update_funding_proxy(sym, progress=lambda m: progress(done / steps, m))
        done += 1
        added[f"{sym} funding (Bitget)"] = data.update_funding(sym, progress=lambda m: progress(done / steps, m))
        done += 1
    progress(1.0, "Data is up to date", force=True)
    return {"kind": "data", "added": added}


def _common(config: dict, tuning: bool = False):
    strat = get_strategy(config["strategy"])
    market = load_market(config["symbols"], config["tf"])
    cfg = bt_config(config["cfg"])
    base = {**strat.default_params, **config.get("params", {})}
    warm = strat.warmup_bars(base)
    if tuning:                          # every parameter set in the grid must have settled indicators too
        grid = config.get("grid") or strat.grid_for(base)
        warm = max([warm] + [strat.warmup_bars(c) for c in expand_grid(strat, base, grid)] if grid else [warm])
    d_start, d_end = default_window(market, warm)
    start = max(config.get("start_ms") or d_start, d_start)
    end = min(config.get("end_ms") or d_end, d_end)
    if end - start < 90 * 24 * 3600 * 1000:
        raise ValueError("The chosen period is shorter than 90 days; results would be meaningless.")
    return strat, market, cfg, start, end


def run_audit(config: dict, progress: Progress) -> dict:
    strat, market, cfg, start, end = _common(config)
    params = {**strat.default_params, **config.get("params", {})}
    progress(0.05, "Simulating the strategy")
    block = evaluate(market, strat, params, cfg, start, end, config.get("n_random", 200),
                     progress=progress.sub(0.15, 0.95))
    checks = make_checks(block)
    return {"kind": "audit", "config": config, "cfg": cfg.to_dict(), "data": _data_info(market),
            "block": block, "checks": checks, "verdict": make_verdict(checks, block["summary"]["trades"])}


def run_oos(config: dict, progress: Progress) -> dict:
    strat, market, cfg, start, end = _common(config, tuning=True)
    tf = market.tf_ms
    split = start + int((end - start) * config.get("split_pct", 60) / 100)
    split = (split // tf) * tf
    base = {**strat.default_params, **config.get("params", {})}
    grid = config.get("grid") or strat.grid_for(base)
    if not grid:
        raise ValueError(f"{strat.label} has no tuning grid defined")
    combos = expand_grid(strat, base, grid)
    progress(0.03, f"Tuning {len(combos)} parameter sets on the training period only")
    windows = {"train": (start, split), "test": (split, end)}
    results = run_grid(market, strat, combos, cfg, windows, progress.sub(0.05, 0.62))
    diag = select_and_diagnose(results, config.get("min_trades", 20))

    out = {"kind": "oos", "config": config, "cfg": cfg.to_dict(), "data": _data_info(market),
           "split": {"train_start": start, "split": split, "test_end": end}, "diag": diag,
           "default_params": base}
    sel = diag["selected"]
    if sel is None:
        checks = [_chk("Tuning found something to test", "fail",
                       f"No parameter set made at least {diag['min_trades']} trades in the training period, "
                       f"so nothing could be selected honestly.", True)]
        out.update(train=None, test=None, checks=checks, verdict=make_verdict(checks, 0))
        return out

    progress(0.64, "Measuring the chosen parameters on the training period")
    train = evaluate(market, strat, sel["params"], cfg, start, split, config.get("n_random", 200) // 2,
                     progress=progress.sub(0.64, 0.78), seed=11)
    progress(0.80, "Measuring the same parameters ONCE on the unseen test period")
    test = evaluate(market, strat, sel["params"], cfg, split, end, config.get("n_random", 200),
                    progress=progress.sub(0.80, 0.95), seed=13)
    progress(0.96, "Measuring your original (untuned) parameters on the test period for comparison")
    untuned = evaluate(market, strat, base, cfg, split, end, 0, stress=False)

    oos = {"train_mean_r": train["summary"]["mean_r"], "test_mean_r": test["summary"]["mean_r"],
           "spearman": diag["spearman"], "luck_t": diag["expected_best_t_by_luck"],
           "train_t": train["summary"]["t_stat"], "n_tried": diag["n_eligible"]}
    checks = make_checks(test, oos)
    out.update(train=train, test=test, untuned_test=untuned, checks=checks,
               verdict=make_verdict(checks, test["summary"]["trades"]))
    return out


def brief(res: dict) -> dict:
    """Tiny summary stored next to the big result, for the run list."""
    if res["kind"] == "data":
        return {"added": int(sum(res["added"].values()))}
    b = res.get("test") or res.get("block")
    out = {"level": res["verdict"]["level"], "headline": res["verdict"]["headline"]}
    if b is not None:
        s = b["summary"]
        out.update(trades=s["trades"], return_pct=s["return_pct"], bh_return_pct=b["bh"]["summary"]["return_pct"],
                   years=s["years"])
    if res.get("train"):
        out["train_return_pct"] = res["train"]["summary"]["return_pct"]
    return out


JOBS = {"data": run_data, "audit": run_audit, "oos": run_oos}
