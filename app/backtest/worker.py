"""Runs ONE backtest job in its own process:  python -m app.backtest.worker <run_id>

Started by the web app. Running it separately keeps a long job from slowing the
website, and a crash here can't take the site or the live paper bot down.
"""
import json
import os
import sys
import traceback

from app import db as database
from app.backtest import runs
from app.db import db


def main(run_id: int) -> None:
    try:
        os.nice(5)                      # be polite: the live bot and website come first
    except OSError:
        pass
    database.init()
    with db() as c:
        row = c.execute("SELECT kind, config FROM bt_runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        sys.exit(f"run {run_id} not found")
    progress = runs.Progress(run_id)
    progress(0.0, "Starting", force=True)
    try:
        result = runs.JOBS[row["kind"]](json.loads(row["config"]), progress)
        payload = runs.to_json(result)
        with db() as c:
            c.execute("UPDATE bt_runs SET status='done', progress=1, message='Done', result=?, brief=? WHERE id=?",
                      (payload, runs.to_json(runs.brief(result)), run_id))
    except Exception as e:              # record the failure so the page can show it
        traceback.print_exc()
        with db() as c:
            c.execute("UPDATE bt_runs SET status='error', message=? WHERE id=?",
                      (f"{type(e).__name__}: {e}"[:500], run_id))


if __name__ == "__main__":
    main(int(sys.argv[1]))
