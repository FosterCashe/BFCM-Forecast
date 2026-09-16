"""Poll model_runs for status='queued', run the pipeline, repeat every
WORKER_POLL_INTERVAL seconds (default 30, per worker/README.md).
"""
import logging
import os
import signal
import time

import psycopg2.extras

from worker.db import get_conn
from worker.pipeline import run_model_run

POLL_INTERVAL = float(os.environ.get("WORKER_POLL_INTERVAL", "30"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s worker %(levelname)s %(message)s")
log = logging.getLogger("worker.loop")

_stop = False


def _handle_stop(signum, frame):
    global _stop
    log.info("received signal %s, stopping after the current cycle", signum)
    _stop = True


def _claim_next_run(conn):
    """SELECT ... FOR UPDATE SKIP LOCKED so multiple worker replicas never
    double-claim the same run."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            select run_id from model_runs
            where status = 'queued'
            order by started_at
            for update skip locked
            limit 1
            """
        )
        row = cur.fetchone()
        if row is None:
            conn.commit()
            return None
        cur.execute("update model_runs set status = 'running' where run_id = %s", (row["run_id"],))
    conn.commit()
    return row["run_id"]


def run_once():
    """Claim and process at most one queued run. Returns True if it found work."""
    with get_conn() as conn:
        run_id = _claim_next_run(conn)
    if run_id is None:
        return False
    log.info("processing run %s", run_id)
    with get_conn() as conn:
        try:
            run_model_run(conn, run_id)
            log.info("run %s done", run_id)
        except Exception:
            log.exception("run %s failed", run_id)
    return True


def main():
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    log.info("worker loop starting, polling every %ss", POLL_INTERVAL)
    while not _stop:
        found = run_once()
        if not found and not _stop:
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
