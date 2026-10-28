# Crypto Trading Lab

A paper-trading lab for crypto perpetual futures: a **cost-aware backtester**, a set of **swappable strategies**, and
**live paper bots** that trade them forward on real exchange prices, with the evidence for each strategy tracked
through four gates before you trust it. It is built around one idea: most trading backtests flatter the strategy, so
the tool is designed to make that hard (costs included, no lookahead, out-of-sample tests, random-entry baselines).

> **It never places real orders.** There is no order-placing code anywhere in this repo. It only reads public
> Bitget market data, and no exchange or wallet key is ever needed or asked for. Everything is paper money.
> Nothing here is financial advice.

**Stack:** Python 3, FastAPI, Jinja2 templates, SQLite, pandas and NumPy. No JavaScript build step.

## Quick start

```bash
git clone https://github.com/mihir13/crypto-trading-lab.git
cd crypto-trading-lab
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8420
```

Open http://127.0.0.1:8420. The database (`data/lab.db`) is created on first run. Download candles from the
Backtest Lab page before running your first backtest. There is no login, so keep the port off the public internet.

Run the tests (111 hand-checked cases, including "no lookahead" guards):

```bash
venv/bin/python -m unittest discover -s tests -t .
```

## How it works

A **strategy** is a saved configuration: a strategy type, its parameters, pairs, timeframe and
risk/cost settings. Two things run from it:

- **Backtests** replay it over stored history with the cost-aware simulator. Run an audit or an
  out-of-sample test straight from the strategy's page; every run is linked back to the strategy.
- **Live paper bots** trade it forward in real time. Each bot has its **own account** (balance, trades,
  kill switch, event log), so any number can run in parallel and be compared fairly. A bot keeps a
  snapshot of the config it started with; editing the strategy later never changes a running bot.

## Pages

| Page | What it does |
|---|---|
| Dashboard | Every strategy's evidence pipeline at a glance: backtest, out-of-sample, live, and the next step |
| Strategies | Create/edit/duplicate/archive strategies; run backtests or start a paper bot from one |
| Live Bots | All paper bots side by side; a bot page shows its equity, positions, trades, config snapshot and log |
| Backtest Lab | Stored Bitget candles, audit and out-of-sample tests, prefilled from a saved strategy |
| Analysis | Compare strategies: scorecards, equity curves, per-pair breakdown, honesty checks, and the backtest-vs-live replay |

## The evidence pipeline

Each strategy is scored on four gates, each needing the one before, and each counted only if it was
obtained with the strategy's **current** settings (a backtest or bot from older settings says nothing
about today's): 1 backtested, 2 passes the honesty checks and made money after costs, 3 the same on the
unseen out-of-sample period, 4 a paper bot with at least 30 closed trades that is profitable. Logic in
`app/analysis.py`.

**Replay** (`/analysis/replay/<bot id>`) runs the backtest simulator over exactly the period a bot has
been live, with its own settings, and matches trades one to one: the gap between backtest and live is
real-world friction (late fills, missed signals) the backtest doesn't model.

## Layout

```
app/
  main.py               web routes (FastAPI)
  db.py                 SQLite schema + helpers (data/lab.db)
  analysis.py           evidence stages, settings matching, backtest-vs-live replay
  strategy_store.py     saved strategies, paper bots, and the one-time migration of the original single bot
  exchange/bitget.py    public market data only
  strategies/           swappable strategies (base.py = the contract)
  futures/engine.py     paper engine + background scheduler (any number of independent bots)
  backtest/             candle-by-candle simulator + honesty checks
  templates/, static/   pages and stylesheet
tests/                  hand-calculated checks of the engine and "no lookahead", pagination, filters
deploy/crypto-trading-lab.service   example systemd unit
docs/
  futures-strategy-protocol.md   how a strategy search is pre-registered before anything is tested
```

## Look

One black-and-yellow, exchange-style theme with the layout of Claude's own app: a quiet left sidebar, one centred
reading column, flat rounded surfaces, sentence-case labels and generous spacing. Colours, the font and
the layout tokens (`--r-lg`, `--r-md`, `--col`) are variables in the `:root` block at the top of
`app/static/style.css`. A price strip at the top shows last price and 24h change for every pair the
strategies trade (public Bitget data, `/api/ticker`).

## Backtest Lab in brief

- **Audit**: runs a strategy as configured over the whole history. No tuning.
- **Out-of-sample**: tunes parameters on the early part of history only, then measures the winner once on a later period it never saw.
- Every result is compared with buy-and-hold and random entries, then run through honesty checks.
- Jobs run in a separate process (`python -m app.backtest.worker <run_id>`) so they can't slow the site or the live bot.

## Pairs, plain English, and charts

- **Pairs:** a strategy can trade up to 12 of any USDT perpetual Bitget lists. The strategy form has a
  search box over the live list (`/api/pairs`, cached ten minutes) plus one-click popular coins; a typed
  pair that Bitget doesn't list is rejected. New coins need their candles downloaded before a backtest
  (one button on the strategy page), and a recently listed coin shortens the backtest window to when every
  pair has data.
- **Plain English:** each strategy page opens with a card explaining what it does, when it buys and sells,
  how much it risks and what its tests found, written from its real settings and results
  (`app/explain.py`, and each strategy's `plain()`). A glossary (`/glossary`) defines the jargon.
- **Charts:** every pair name links to its TradingView chart (`BITGET:<pair>.P`), opened in dark mode.

## Building a strategy in the UI (Custom rules)

Strategies > New strategy > type "Custom rules". Switch on any of: uptrend (close above an EMA), an RSI
condition (dip or momentum), a breakout of the previous N-candle high, an EMA cross. All switched-on
entry conditions must hold on the same candle. Stops and targets are ATR-based, and you can add
exits (channel low, RSI level). A rule set is just a parameter set, so it is saved, backtested,
tuned out-of-sample and run live like any hand-written strategy, with the same no-lookahead
tests (`tests/test_backtest.py`, `tests/test_rules.py`). No code is pasted in anywhere.

## Adding a strategy in code

1. Subclass `Strategy` in `app/strategies/` and implement `compute()` (fast, all candles at once) or just `signal()`/`exit_signal()` (slow but works everywhere). See `ema_rsi.py`. Row *i* of `compute()` may only use candles up to *i*.
2. Register it in `app/strategies/__init__.py`. Its `default_params` become its settings automatically.
3. Set `warmup_bars()` if an indicator needs long history (a 200 EMA needs ~600 candles to settle).
4. Optionally add `param_grid`, `is_valid()`, `explain()` (the signal funnel) and `context()` (its regime filter).
5. Add its `LOOSE` parameters to `tests/test_backtest.py`.

Strategy search rule of thumb: write down what counts as success (windows, parameters, pass bar) BEFORE running anything, keep a final holdout period you look at once, and count every parameter set you try. See `docs/futures-strategy-protocol.md` for a worked example.

## Tests

```
python -m unittest discover -s tests -t .
```

## Routes

FastAPI matches routes in registration order: register static paths (e.g. `/backtest/data`) before any `/{param}` route that could swallow them.

## Secrets

Copy `.env.example` to `.env` (git-ignored). The database in `data/` is also git-ignored. Use read-only keys for any data provider.

## Running it as a service

`deploy/crypto-trading-lab.service` is an example systemd unit. Edit the user and paths, copy it to
`/etc/systemd/system/`, then `systemctl enable --now crypto-trading-lab`. Because there is no login, bind it to
`127.0.0.1` or a private network (LAN or a VPN such as Tailscale).

## About the research docs

`docs/futures-strategy-protocol.md` describes the method the lab is built to support: write down the candidate
strategies, the data windows and the pass/fail bar **before** running anything, keep a final holdout period you open
only once, and count every parameter set you try. It contains no results and no recommendation to trade anything.

## License

MIT. See `LICENSE`.
