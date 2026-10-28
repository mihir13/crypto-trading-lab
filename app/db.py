"""SQLite access. One small database file holds everything (settings,
paper trades, event log), so backups are just a file copy.

WAL mode lets the web server read while the bot thread writes.
"""
import os
import sqlite3
from contextlib import contextmanager

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("LAB_DB") or os.path.join(_HERE, "..", "data", "lab.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL              -- JSON
);

-- Saved strategy configurations: a strategy type plus its parameters, pairs, timeframe and
-- risk/cost settings. Backtests and paper bots are both run FROM one of these.
CREATE TABLE IF NOT EXISTS strategies (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    type      TEXT NOT NULL,             -- key into app.strategies.STRATEGIES
    params    TEXT NOT NULL,             -- JSON
    pairs     TEXT NOT NULL,             -- JSON list
    timeframe TEXT NOT NULL,
    risk      TEXT NOT NULL,             -- JSON: capital, risk_pct, leverage, fees, ...
    notes     TEXT NOT NULL DEFAULT '',
    created   TEXT NOT NULL,
    archived  INTEGER NOT NULL DEFAULT 0
);

-- A paper bot: one deployment of a strategy with its OWN account. `config` is a snapshot taken
-- when the bot was started, so editing the strategy later never changes a running bot.
CREATE TABLE IF NOT EXISTS bots (
    id          INTEGER PRIMARY KEY,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    name        TEXT NOT NULL,
    config      TEXT NOT NULL,           -- JSON snapshot
    status      TEXT NOT NULL,           -- running | paused
    created     TEXT NOT NULL,
    started_at  TEXT NOT NULL
);

-- Per-bot runtime state (balance, kill switch, last scan, marks, ...).
CREATE TABLE IF NOT EXISTS bot_state (
    bot_id INTEGER NOT NULL REFERENCES bots(id) ON DELETE CASCADE,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (bot_id, key)
);

-- Legacy: the original single bot's state. Copied into bot_state by strategy_store.migrate_legacy().
CREATE TABLE IF NOT EXISTS futures_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS futures_trades (
    id            INTEGER PRIMARY KEY,
    symbol        TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    status        TEXT NOT NULL,     -- open | closed
    entry_time    TEXT NOT NULL,
    entry_price   REAL NOT NULL,
    stop          REAL NOT NULL,
    target        REAL NOT NULL,
    units         REAL NOT NULL,
    notional      REAL NOT NULL,
    margin        REAL NOT NULL,
    fee_open      REAL NOT NULL,
    signal_time   TEXT,
    exit_time     TEXT,
    exit_price    REAL,
    outcome       TEXT,
    gross_pnl     REAL,
    fee_close     REAL,
    funding       REAL,
    net_pnl       REAL,
    balance_after REAL
);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY,
    ts      TEXT NOT NULL,
    source  TEXT NOT NULL,
    level   TEXT NOT NULL,
    message TEXT NOT NULL
);

-- Historical market data for the Backtest Lab (public Bitget data, closed candles only).
-- ts is the candle OPEN time in milliseconds (UTC).
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    tf     TEXT NOT NULL,
    ts     INTEGER NOT NULL,
    open   REAL NOT NULL,
    high   REAL NOT NULL,
    low    REAL NOT NULL,
    close  REAL NOT NULL,
    vol    REAL NOT NULL,
    PRIMARY KEY (symbol, tf, ts)
) WITHOUT ROWID;

-- Real historical funding rates (fraction per funding interval; positive = longs pay shorts).
-- src = 'bitget' (real) or 'binance' (proxy for dates older than Bitget keeps, ~90 days).
CREATE TABLE IF NOT EXISTS funding (
    symbol TEXT NOT NULL,
    ts     INTEGER NOT NULL,
    rate   REAL NOT NULL,
    src    TEXT NOT NULL DEFAULT 'bitget',
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;

-- One row per background job: data downloads and backtests.
CREATE TABLE IF NOT EXISTS bt_runs (
    id       INTEGER PRIMARY KEY,
    created  TEXT NOT NULL,
    strategy_id INTEGER,               -- the saved strategy it tested (NULL for data jobs / ad-hoc runs)
    kind     TEXT NOT NULL,            -- data | audit | oos
    status   TEXT NOT NULL,            -- queued | running | done | error
    progress REAL NOT NULL DEFAULT 0,  -- 0..1
    message  TEXT NOT NULL DEFAULT '',
    config   TEXT NOT NULL,            -- JSON
    result   TEXT,                     -- JSON, filled when done
    brief    TEXT                      -- small JSON summary for list pages
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db():
    """`with db() as c:` gives a connection that commits on success."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    with db() as c:
        c.executescript(SCHEMA)
        cols = [r["name"] for r in c.execute("PRAGMA table_info(funding)")]
        run_cols = [r["name"] for r in c.execute("PRAGMA table_info(bt_runs)")]
        for table in ("futures_trades", "events"):        # per-bot rows; NULL = the legacy single bot
            if "bot_id" not in [r["name"] for r in c.execute(f"PRAGMA table_info({table})")]:
                c.execute(f"ALTER TABLE {table} ADD COLUMN bot_id INTEGER")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_bot ON futures_trades(bot_id, status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_events_bot ON events(bot_id, id)")
        if "strategy_id" not in run_cols:
            c.execute("ALTER TABLE bt_runs ADD COLUMN strategy_id INTEGER")
        if "brief" not in run_cols:
            c.execute("ALTER TABLE bt_runs ADD COLUMN brief TEXT")
        if "src" not in cols:                     # tables created before the column existed
            c.execute("ALTER TABLE funding ADD COLUMN src TEXT NOT NULL DEFAULT 'bitget'")
