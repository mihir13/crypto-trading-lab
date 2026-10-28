"""Plain-English explanations of a strategy for someone who doesn't know the jargon: what it does,
how much it risks, and what its tests say, all generated from its real settings and results."""
from app import analysis as an
from app.strategies import STRATEGIES

GLOSSARY = {
    "paper": "Paper trading: simulated trades with pretend money at real market prices. Nothing is ever really bought or sold.",
    "perp": "Perpetual futures: contracts that follow a coin's price with no expiry date, traded with leverage. This site uses Bitget's.",
    "candle": "Candle: one slice of price history (for example 4 hours). It records where the price opened, its highest and lowest points, and where it closed.",
    "ema": "EMA (exponential moving average): the average price over the last N candles, giving recent candles more weight. Price above it suggests an uptrend; below suggests a downtrend.",
    "rsi": "RSI (relative strength index): a 0-100 gauge of how strong recent moves have been. Low readings (below about 30-40) mean the coin has dropped fast; high readings (above about 70) mean it has run up fast.",
    "atr": "ATR (average true range): how far the price typically moves in one candle. Stops are set in ATR units so a wild coin gets more room than a calm one.",
    "stop": "Stop-loss: a preset exit price below the entry. If the price falls to it, the trade is closed to cap the loss.",
    "target": "Profit target: a preset exit price above the entry where the trade is closed to lock in the gain. '2× the amount risked' means aiming to win twice what a stop-out would lose.",
    "volume": "Volume: how much of the coin traded during a candle. High volume means many participants are behind a move.",
    "breakout": "Breakout: the price rising above its recent highest point, often read as buyers taking control.",
    "leverage": "Leverage: trading a position larger than your own money by borrowing. At 5x, $100 controls $500 of coin, which magnifies both gains and losses.",
    "drawdown": "Drawdown: the fall from the account's highest point to a later low. A worst drawdown of -30% means that at some point it was 30% below its peak.",
    "backtest": "Backtest: replaying a strategy over past prices to see what it would have done, including fees and slippage.",
    "oos": "Out-of-sample test: tuning a strategy on older data, then checking it once on later data it never saw. It's the honest check against a strategy that only fits the past by luck.",
    "funding": "Funding rate: a small periodic payment between long and short traders on perpetual futures. It can cost or earn money while a trade is open.",
    "slippage": "Slippage: the gap between the price you expect and the price you actually get when trading.",
}
ALWAYS = ["paper", "perp", "leverage", "slippage", "funding"]
_LEVEL = {"none": "found no reliable edge", "weak": "found weak evidence, not enough to trust",
          "moderate": "found some evidence, worth forward-testing"}


def define(key: str) -> tuple[str, str]:
    """(term, definition) for a glossary key."""
    term, _, text = GLOSSARY[key].partition(": ")
    return term, text


def risk_lines(s: dict) -> list[str]:
    r, cap = s["risk"], s["risk"]["starting_capital"]
    per_trade = cap * r["risk_pct"] / 100
    coins = ", ".join(p.removesuffix("USDT") for p in s["pairs"])
    return [
        f"Trades {coins} on {s['timeframe']} candles, checking the market every {r['scan_minutes']} minutes.",
        f"Each trade risks about {r['risk_pct']:g}% of the account ({'$%.2f' % per_trade} of ${cap:,.0f}) if it hits its stop-loss.",
        f"Uses {r['leverage']}x leverage, but never puts more than {r['max_margin_pct']:g}% of the account into one trade, "
        f"and holds at most {r['max_simultaneous']} trades at once.",
        f"Closes any trade after {r['max_hold_hours']} hours, and stops trading for the day if the account falls {r['daily_loss_limit_pct']:g}%.",
        f"Assumes real-world costs: a {r['taker_fee'] * 100:.3f}% fee on every fill and {r['slippage_pct']:g}% slippage against it.",
    ]


def track_lines(row: dict, series: dict | None) -> list[str]:
    """What the tests found, in sentences. `row` is analysis.strategy_row(); `series` the audit's
    analysis.backtest_series() (for the worst drop and win rate), or None."""
    out = []
    a, o, live = row["audit"], row["oos"], row["live"]
    if a:
        s = (series or {}).get("summary", {})
        line = (f"Replayed over {a.get('years', 0):.1f} years of past prices it made {a['trades']} trades and returned "
                f"{a['return_pct']:+.0f}% overall")
        if a.get("bh_return_pct") is not None:
            line += f", while simply buying and holding the same coins would have returned {a['bh_return_pct']:+.0f}%"
        out.append(line + ".")
        if s.get("max_dd_pct") is not None:
            extra = f"Its worst fall from a peak was {abs(s['max_dd_pct']):.0f}%"
            if s.get("win_rate") is not None:
                extra += f", and {s['win_rate']:.0f}% of its trades were winners"
            out.append(extra + ".")
        out.append(f"The honesty checks {_LEVEL.get(a.get('level'), 'gave no verdict')}.")
    else:
        out.append("Not backtested with its current settings yet. Run a backtest from this page to see how it would have done." if not row["stale"]
                   else "It has been backtested before, but its settings have changed since. Run a new backtest to see how these settings would have done.")
    if o:
        out.append(f"On a later period it hadn't been tuned on, it returned {o['return_pct']:+.0f}% ({o['trades']} trades); "
                   f"that test {_LEVEL.get(o.get('level'), 'gave no verdict')}.")
    if live:
        few = live["trades"] < an.MIN_LIVE_TRADES
        out.append(f"Trading live on paper for {live['days']:.0f} days it has made {live['trades']} trades and returned {live['return_pct']:+.1f}%"
                   + (f". That's too few trades to judge (about {an.MIN_LIVE_TRADES} are needed)." if few else "."))
    return out


def summary(s: dict, row: dict | None = None, series: dict | None = None) -> dict:
    """Everything the 'In plain English' card shows for a saved strategy."""
    strat = STRATEGIES[s["type"]]
    p = strat.plain(s["params"])
    terms = list(dict.fromkeys(p["terms"] + ALWAYS))
    return {**p, "risk": risk_lines(s), "track": track_lines(row, series) if row else [],
            "next": (row["next"], row["tone"]) if row else None,
            "glossary": [define(t) for t in terms if t in GLOSSARY]}
