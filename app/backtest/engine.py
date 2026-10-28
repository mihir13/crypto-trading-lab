"""Candle-by-candle backtest engine.

It runs the SAME strategy code and the SAME cost model as the live paper bot
(app/futures/engine.py), only over history. Every rule below is chosen so that a
backtest is more likely to understate than overstate what live trading would do.

Timeline for one candle k (a candle covers [open_k, open_k + timeframe)):
    signals are computed on the CLOSE of candle k-1, so they only use data that
    existed before candle k opened (no lookahead). Then, in this order:
      a) market exits decided earlier (time stop, strategy exit) fill at the OPEN of k
      b) new entries fill at the OPEN of k
      c) stop / target are checked against candle k's LOW and HIGH
      d) equity is marked at the CLOSE of k

Pessimistic rules:
  * every entry and every market exit fills `slippage_pct` worse than the price
  * every fill pays the taker fee
  * if one candle touches both stop and target, the STOP is assumed to hit first
  * a stop that gaps through fills at the worse of stop and the candle's open
  * a target is only credited at the target price, never at a better gapped price
  * funding uses real historical rates (Binance proxy before Bitget's own 90 days)
  * isolated-margin liquidation is modelled
What it does NOT model (so real results can still be worse): order-book depth for
larger sizes, partial fills, exchange downtime, API latency, and stop orders being
skipped in fast markets. See the assumptions box on the Backtest page.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000


@dataclass
class BtConfig:
    capital: float = 100.0
    risk_pct: float = 3.0            # % of balance risked per trade (stop distance sizing)
    leverage: float = 5.0
    max_margin_pct: float = 40.0     # max margin on one trade, % of balance
    max_simultaneous: int = 3
    taker_fee: float = 0.0006        # per side, applied to every fill
    slippage_pct: float = 0.05       # per market fill, against you
    max_hold_hours: float = 72.0
    max_funding_abs: float = 0.001   # skip entries when |funding| is above this (as live)
    maint_margin: float = 0.005      # approx. maintenance margin for the liquidation price
    fallback_funding: float = 0.0001 # per 8h, used only when no funding history exists
    funding_worst_case: bool = False # charge |rate| even when it would have been received
    entry_delay_bars: int = 0        # stress test: enter this many candles after the signal

    def to_dict(self) -> dict:
        return asdict(self)


class Market:
    """Candles for several symbols laid out on one shared timeline of open times."""

    def __init__(self, candles: dict[str, pd.DataFrame], funding: dict, tf_ms: int):
        self.symbols = list(candles)
        self.tf_ms = tf_ms
        self.frames = candles
        self.funding = funding                       # symbol -> (ts_ms array, rate array)
        self.T = np.unique(np.concatenate([df["ts_ms"].to_numpy(dtype=np.int64) for df in candles.values()]))
        n = len(self.T)
        self.idx, self.O, self.H, self.L, self.C, self.V = {}, {}, {}, {}, {}, {}
        for sym, df in candles.items():
            idx = np.searchsorted(self.T, df["ts_ms"].to_numpy(dtype=np.int64))
            self.idx[sym] = idx
            for store, col in ((self.O, "open"), (self.H, "high"), (self.L, "low"), (self.C, "close")):
                a = np.full(n, np.nan)
                a[idx] = df[col].to_numpy(dtype=float)
                store[sym] = a
            v = np.zeros(n, dtype=bool)
            v[idx] = True
            self.V[sym] = v

    def signals(self, strategy, params: dict) -> dict:
        """Run the strategy over every symbol and align its output to the shared timeline."""
        n = len(self.T)
        out = {}
        for sym, df in self.frames.items():
            c = strategy.compute(df, params)
            E = np.zeros(n, dtype=bool)
            X = np.zeros(n, dtype=bool)
            C = np.zeros(n, dtype=bool)                  # regime filter: where entries are even considered
            S = np.full(n, np.nan)
            G = np.full(n, np.nan)
            E[self.idx[sym]] = c["enter"].to_numpy(dtype=bool)
            X[self.idx[sym]] = c["exit"].to_numpy(dtype=bool)
            C[self.idx[sym]] = np.asarray(strategy.context(df, params), dtype=bool)
            S[self.idx[sym]] = c["stop"].to_numpy(dtype=float)
            G[self.idx[sym]] = c["target"].to_numpy(dtype=float)
            out[sym] = {"E": E, "X": X, "S": S, "G": G, "C": C}
        return out

    def window_bounds(self, start_ms: int, end_ms: int) -> tuple[int, int]:
        """Index range [k0, k1] of candles that OPEN inside [start_ms, end_ms)."""
        k0 = max(int(np.searchsorted(self.T, start_ms, side="left")), 1)
        k1 = int(np.searchsorted(self.T, end_ms, side="left")) - 1
        return k0, k1


@dataclass
class SimResult:
    trades: list
    eq_ts: np.ndarray               # ms timestamps (candle closes)
    eq: np.ndarray                  # mark-to-market equity at each candle close
    final_balance: float
    bars: int
    exposure: float                 # share of candles with at least one open position
    signals: int                    # entry signals seen inside the window
    skipped: dict                   # why some signals were not traded
    flat_symbol_bars: int           # symbol-candles spent without a position (for the random baseline)
    total_symbol_bars: int


def _funding_cost(fund, t0: int, t1: int, notional: float, cfg: BtConfig) -> float:
    """Funding paid (positive) or received (negative) by a long held from t0 to t1."""
    ts, rate = fund if fund is not None else (np.array([], dtype=np.int64), np.array([]))
    if ts.size == 0 or t0 < ts[0]:
        return notional * cfg.fallback_funding * max(t1 - t0, 0) / (8 * HOUR_MS)
    lo = int(np.searchsorted(ts, t0, side="right"))     # funding times after entry ...
    hi = int(np.searchsorted(ts, t1, side="right"))     # ... up to and including exit
    r = rate[lo:hi]
    total = float(np.abs(r).sum() if cfg.funding_worst_case else r.sum())
    return notional * total


def simulate(market: Market, sigs: dict, cfg: BtConfig, start_ms: int, end_ms: int) -> SimResult:
    T, tf = market.T, market.tf_ms
    k0, k1 = market.window_bounds(start_ms, end_ms)
    if k1 < k0:
        raise ValueError("The chosen window contains no candles")
    slip = cfg.slippage_pct / 100.0
    max_hold_ms = cfg.max_hold_hours * HOUR_MS
    delay = cfg.entry_delay_bars
    syms = market.symbols

    balance = cfg.capital
    open_pos: dict = {}
    trades: list = []
    eq_ts, eq_val = [], []
    bars_in_market = 0
    flat_bars = 0
    signal_count = 0
    skipped = {"no_slot": 0, "margin": 0, "funding": 0, "price_range": 0, "ruined": 0}
    last_close = {s: np.nan for s in syms}

    def close_position(sym, price, exit_ts, outcome):
        nonlocal balance
        p = open_pos.pop(sym)
        units = p["units"]
        if outcome == "LIQUIDATED":
            gross, fee_close = -p["margin"], 0.0          # isolated margin is lost entirely
        else:
            gross = (price - p["entry"]) * units
            fee_close = price * units * cfg.taker_fee
        funding = _funding_cost(market.funding.get(sym), p["entry_ts"], exit_ts, p["notional"], cfg)
        net = gross - p["fee_open"] - fee_close - funding
        balance += gross - fee_close - funding            # fee_open was deducted at entry
        trades.append({
            "symbol": sym, "entry_ts": p["entry_ts"], "exit_ts": exit_ts,
            "entry": p["entry"], "exit": price, "units": units, "notional": p["notional"],
            "fee_open": p["fee_open"], "fee_close": fee_close, "funding": funding,
            "gross": gross, "net": net, "outcome": outcome,
            "risk_amount": p["risk_amount"], "r": net / p["risk_amount"] if p["risk_amount"] else 0.0,
            "hold_h": (exit_ts - p["entry_ts"]) / HOUR_MS,
            "pct_move": (price / p["entry"] - 1) * 100, "balance_after": balance,
        })

    for k in range(k0, k1 + 1):
        t = int(T[k])

        # a) market exits at this candle's open
        for sym in list(open_pos):
            if not market.V[sym][k]:
                continue
            p = open_pos[sym]
            reason = None
            if t - p["entry_ts"] >= max_hold_ms:
                reason = "MAX HOLD"
            elif sigs[sym]["X"][k - 1]:
                reason = "STRATEGY EXIT"
            if reason:
                close_position(sym, market.O[sym][k] * (1 - slip), t, reason)

        # b) entries at this candle's open (decided on the close of an earlier candle)
        sk = k - 1 - delay
        for sym in syms:
            if sk < 0 or not (market.V[sym][k] and market.V[sym][sk] and sigs[sym]["E"][sk]):
                continue
            signal_count += 1
            if sym in open_pos:
                continue                                  # already in this pair (not a lost signal)
            if len(open_pos) >= cfg.max_simultaneous:
                skipped["no_slot"] += 1
                continue
            if balance <= 0:
                skipped["ruined"] += 1
                continue
            fund = market.funding.get(sym)
            if fund is not None and fund[0].size:
                j = int(np.searchsorted(fund[0], t, side="right")) - 1
                if j >= 0 and abs(fund[1][j]) > cfg.max_funding_abs:
                    skipped["funding"] += 1
                    continue
            stop, target = sigs[sym]["S"][sk], sigs[sym]["G"][sk]
            fill = market.O[sym][k] * (1 + slip)
            if not (stop < fill < target):                # already through the levels
                skipped["price_range"] += 1
                continue
            risk_amount = balance * cfg.risk_pct / 100
            units = risk_amount / (fill - stop)
            notional = units * fill
            margin = notional / cfg.leverage
            fee_open = notional * cfg.taker_fee
            available = balance - sum(q["margin"] for q in open_pos.values())
            if margin + fee_open > min(balance * cfg.max_margin_pct / 100, available):
                skipped["margin"] += 1
                continue
            balance -= fee_open                           # opening fee is paid immediately
            open_pos[sym] = {
                "entry": fill, "entry_ts": t, "stop": stop, "target": target, "units": units,
                "notional": notional, "margin": margin, "fee_open": fee_open, "risk_amount": risk_amount,
                "liq": fill * (1 - 1 / cfg.leverage + cfg.maint_margin),
            }

        # c) stop / target against this candle's high and low (stop first: pessimistic)
        for sym in list(open_pos):
            if not market.V[sym][k]:
                continue
            p = open_pos[sym]
            lo, hi, op = market.L[sym][k], market.H[sym][k], market.O[sym][k]
            eff_stop = max(p["stop"], p["liq"])
            if lo <= eff_stop:
                if p["liq"] >= p["stop"]:
                    close_position(sym, p["liq"], t + tf, "LIQUIDATED")
                else:
                    close_position(sym, min(p["stop"], op) * (1 - slip), t + tf, "STOP LOSS")
            elif hi >= p["target"]:
                close_position(sym, p["target"], t + tf, "TAKE PROFIT")

        # d) mark equity at this candle's close
        eq = balance
        for sym, p in open_pos.items():
            c = market.C[sym][k]
            if np.isnan(c):
                c = last_close[sym]
            eq += (c - p["entry"]) * p["units"]
        for s in syms:
            if market.V[s][k]:
                last_close[s] = market.C[s][k]
        eq_ts.append(t + tf)
        eq_val.append(eq)
        if open_pos:
            bars_in_market += 1
        flat_bars += sum(1 for s in syms if market.V[s][k] and s not in open_pos)

    # window over: close anything still open at the final close (costs included)
    for sym in list(open_pos):
        c = last_close[sym]
        close_position(sym, c * (1 - slip), int(T[k1]) + tf, "END OF WINDOW")
    if eq_val:
        eq_val[-1] = balance

    bars = k1 - k0 + 1
    return SimResult(
        trades=trades, eq_ts=np.array(eq_ts, dtype=np.int64), eq=np.array(eq_val, dtype=float),
        final_balance=balance, bars=bars, exposure=bars_in_market / bars if bars else 0.0,
        signals=signal_count, skipped=skipped, flat_symbol_bars=flat_bars,
        total_symbol_bars=sum(int(market.V[s][k0:k1 + 1].sum()) for s in syms),
    )
