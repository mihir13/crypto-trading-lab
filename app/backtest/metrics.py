"""Performance numbers and the two baselines a strategy has to beat.

A strategy's return means nothing on its own. It has to be compared with
(1) simply buying and holding the same coins, and
(2) entering at random with the same exits and costs.
If it can't beat both by a clear margin, the "edge" is probably just the market
going up or the exit rules, not the entry signal.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.backtest.engine import BtConfig, Market, SimResult, simulate

YEAR_MS = 365.25 * 24 * 3600 * 1000


def _f(x):
    """numpy -> plain float (JSON friendly); None stays None."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def max_drawdown_pct(eq: np.ndarray) -> float:
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min() * 100)


def sharpe(eq: np.ndarray, tf_ms: int) -> float | None:
    if len(eq) < 3:
        return None
    r = np.diff(eq) / eq[:-1]
    sd = r.std()
    if sd == 0:
        return None
    return float(r.mean() / sd * np.sqrt(YEAR_MS / tf_ms))


def summarize(sim: SimResult, cfg: BtConfig, tf_ms: int) -> dict:
    trades = sim.trades
    cap = cfg.capital
    eq = sim.eq
    n = len(trades)
    years = (sim.eq_ts[-1] - sim.eq_ts[0] + tf_ms) / YEAR_MS if len(sim.eq_ts) else 0.0
    ret = (sim.final_balance / cap - 1) * 100
    mdd = max_drawdown_pct(eq)

    net = np.array([t["net"] for t in trades])
    R = np.array([t["r"] for t in trades])
    wins, losses = net[net > 0], net[net <= 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    pf = gp / gl if gl > 0 else None

    # longest run of consecutive losing trades
    streak = best_streak = 0
    for x in net:
        streak = streak + 1 if x <= 0 else 0
        best_streak = max(best_streak, streak)

    top3 = float(np.sort(net)[-3:].sum() / gp) if gp > 0 and n >= 3 else None
    mean_r = float(R.mean()) if n else None
    t_stat = float(R.mean() / (R.std(ddof=1) / np.sqrt(n))) if n >= 3 and R.std(ddof=1) > 0 else None

    outcomes: dict = {}
    for t in trades:
        outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1

    return {
        "return_pct": _f(ret), "max_dd_pct": _f(mdd),
        "cagr_pct": _f(((sim.final_balance / cap) ** (1 / years) - 1) * 100) if years >= 0.5 and sim.final_balance > 0 else None,
        "calmar": _f(ret / abs(mdd)) if mdd < 0 else None,
        "sharpe": _f(sharpe(eq, tf_ms)),
        "final_balance": _f(sim.final_balance), "years": _f(years),
        "trades": n, "wins": int((net > 0).sum()), "win_rate": _f((net > 0).mean() * 100) if n else None,
        "profit_factor": _f(pf), "gross_profit": _f(gp), "gross_loss": _f(gl),
        "avg_win": _f(wins.mean()) if len(wins) else None, "avg_loss": _f(losses.mean()) if len(losses) else None,
        "best_trade": _f(net.max()) if n else None, "worst_trade": _f(net.min()) if n else None,
        "mean_r": _f(mean_r), "t_stat": _f(t_stat),
        "top3_share": _f(top3), "loss_streak": best_streak,
        "fees": _f(sum(t["fee_open"] + t["fee_close"] for t in trades)),
        "funding": _f(sum(t["funding"] for t in trades)),
        "avg_hold_h": _f(np.mean([t["hold_h"] for t in trades])) if n else None,
        "trades_per_month": _f(n / (years * 12)) if years else None,
        "exposure_pct": _f(sim.exposure * 100), "signals": sim.signals,
        "skipped": sim.skipped, "outcomes": outcomes,
    }


def bootstrap_mean_r(R: list[float], n_boot: int = 4000, seed: int = 1) -> dict | None:
    """95% confidence interval for the average R per trade, by resampling the trades.

    If the interval includes 0, the data cannot distinguish this strategy from luck.
    """
    R = np.asarray(R, dtype=float)
    if len(R) < 8:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(R, size=(n_boot, len(R)), replace=True).mean(axis=1)
    return {"mean": float(R.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "p_le_zero": float((means <= 0).mean())}


# ── baseline 1: buy and hold ─────────────────────────────────────────────────

def buy_and_hold(market: Market, cfg: BtConfig, start_ms: int, end_ms: int) -> dict:
    """Equal-weight, unleveraged, no rebalancing. Pays the same fee and slippage on the
    way in and out, so the comparison is fair."""
    k0, k1 = market.window_bounds(start_ms, end_ms)
    slip, fee = cfg.slippage_pct / 100, cfg.taker_fee
    syms = [s for s in market.symbols if market.V[s][k0]]
    if not syms:
        raise ValueError("No symbol has data at the start of the window")
    alloc = cfg.capital / len(syms)
    T = market.T
    eq = np.zeros(k1 - k0 + 1)
    per_symbol = {}
    final = 0.0
    for s in syms:
        fill = market.O[s][k0] * (1 + slip)
        units = alloc / (1 + fee) / fill
        close = pd.Series(market.C[s][k0:k1 + 1]).ffill().to_numpy()
        curve = units * close
        eq += curve
        exit_val = units * close[-1] * (1 - slip) * (1 - fee)
        final += exit_val
        solo = np.concatenate([[alloc], curve])
        per_symbol[s] = {"return_pct": _f((exit_val / alloc - 1) * 100),
                         "max_dd_pct": _f(max_drawdown_pct(solo))}
    eq[-1] = final
    ts = T[k0:k1 + 1] + market.tf_ms
    return {
        "eq_ts": ts, "eq": eq, "per_symbol": per_symbol,
        "summary": {"return_pct": _f((final / cfg.capital - 1) * 100),
                    "max_dd_pct": _f(max_drawdown_pct(np.concatenate([[cfg.capital], eq]))),
                    "sharpe": _f(sharpe(eq, market.tf_ms)), "final_balance": _f(final)},
    }


# ── baseline 2: random entries with identical exits and costs ────────────────

def random_baseline(market: Market, sigs: dict, cfg: BtConfig, start_ms: int, end_ms: int,
                    strategy_summary: dict, sim: SimResult, n_sims: int = 200, seed: int = 7,
                    progress=None, regime: bool = False) -> dict | None:
    """Replace ONLY the entry signal with coin flips (same stops/targets/exits/costs).

    The flip probability is set so the random runs make about as many trades as the
    strategy did. If the real strategy doesn't clearly beat these, its entry signal
    adds nothing; what you're seeing is the exits, the costs and the market.

    regime=False: coin flips on ANY candle.
    regime=True:  coin flips only on candles inside the strategy's own regime filter
                  (e.g. "above the 200 EMA"). Beating THIS baseline means the trigger adds
                  value beyond simply being in that regime.
    """
    n_trades = strategy_summary["trades"]
    flat = sim.flat_symbol_bars
    if n_sims <= 0 or n_trades < 3 or flat <= 0:
        return None
    p = min(n_trades / flat, 0.5)
    n = len(market.T)
    ok = {s: ~np.isnan(sigs[s]["S"]) & market.V[s] for s in market.symbols}
    if regime:
        k0, k1 = market.window_bounds(start_ms, end_ms)
        all_bars = sum(int(ok[s][k0:k1 + 1].sum()) for s in market.symbols)
        ok = {s: ok[s] & sigs[s]["C"] for s in market.symbols}
        in_regime = sum(int(ok[s][k0:k1 + 1].sum()) for s in market.symbols)
        if in_regime == 0 or all_bars == 0:
            return None
        p = min(p * all_bars / in_regime, 0.5)      # keep the expected number of trades the same
    runs = []
    for i in range(n_sims):
        rng = np.random.default_rng([seed, i])
        rs = {s: {"E": (rng.random(n) < p) & ok[s], "X": sigs[s]["X"], "S": sigs[s]["S"], "G": sigs[s]["G"]}
              for s in market.symbols}
        r = summarize(simulate(market, rs, cfg, start_ms, end_ms), cfg, market.tf_ms)
        runs.append({"return_pct": r["return_pct"], "profit_factor": r["profit_factor"],
                     "mean_r": r["mean_r"], "trades": r["trades"], "max_dd_pct": r["max_dd_pct"]})
        if progress and (i + 1) % 10 == 0:
            progress((i + 1) / n_sims)
    rets = np.array([x["return_pct"] for x in runs])
    mr = np.array([x["mean_r"] if x["mean_r"] is not None else 0.0 for x in runs])
    mine, mine_r = strategy_summary["return_pct"], strategy_summary["mean_r"] or 0.0
    return {
        "n": n_sims, "prob_per_bar": float(p),
        "returns": [float(x) for x in rets],
        "mean_trades": float(np.mean([x["trades"] for x in runs])),
        "median": float(np.median(rets)), "p5": float(np.percentile(rets, 5)), "p95": float(np.percentile(rets, 95)),
        "median_mean_r": float(np.median(mr)),
        # share of random runs the strategy beat (100% = better than every one of them)
        "pctile_return": float((rets < mine).mean() * 100),
        "pctile_mean_r": float((mr < mine_r).mean() * 100),
    }


# ── tables for the page ──────────────────────────────────────────────────────

def downsample(ts: np.ndarray, vals: np.ndarray, max_points: int = 900) -> dict:
    if len(ts) > max_points:
        keep = np.unique(np.linspace(0, len(ts) - 1, max_points).astype(int))
        ts, vals = ts[keep], vals[keep]
    return {"t": [int(x) for x in ts], "v": [round(float(x), 4) for x in vals]}


def by_symbol(trades: list) -> dict:
    out: dict = {}
    for t in trades:
        d = out.setdefault(t["symbol"], {"trades": 0, "wins": 0, "net": 0.0})
        d["trades"] += 1
        d["wins"] += t["net"] > 0
        d["net"] += t["net"]
    return {s: {**d, "net": round(d["net"], 4)} for s, d in out.items()}


def by_quarter(sim: SimResult, bh: dict, capital: float) -> list:
    """Calendar-quarter returns for the strategy and for buy-and-hold, side by side."""
    def q_returns(ts, eq, start):
        s = pd.Series(eq, index=pd.to_datetime(ts, unit="ms"))   # naive UTC (to_period drops tz anyway)
        last = s.groupby(s.index.to_period("Q")).last()
        prev = last.shift(1).fillna(start)
        return (last / prev - 1) * 100, last

    sr, _ = q_returns(sim.eq_ts, sim.eq, capital)
    br, _ = q_returns(bh["eq_ts"], bh["eq"], capital)
    trades = pd.Series([1] * len(sim.trades),
                       index=pd.to_datetime([t["exit_ts"] for t in sim.trades], unit="ms")) \
        if sim.trades else pd.Series(dtype=int)
    tq = trades.groupby(trades.index.to_period("Q")).sum() if len(trades) else pd.Series(dtype=int)
    return [{"q": str(q), "strategy": _f(sr[q]), "hold": _f(br.get(q)), "trades": int(tq.get(q, 0))} for q in sr.index]


def fmt_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")
