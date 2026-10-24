import sqlite3
from contextlib import closing
from datetime import datetime, timedelta

import pandas as pd

import config


def _conn():
    config.DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(config.DB_PATH)


def init_db():
    with closing(_conn()) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS readings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, ts TEXT, total INTEGER, engaged INTEGER,
            partial INTEGER, disengaged INTEGER, class_pct REAL,
            hands INTEGER, phones INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, ts TEXT, etype TEXT, detail TEXT, pct REAL)""")
        c.commit()


def log_reading(session_id, counts, pct):
    with closing(_conn()) as c:
        c.execute("INSERT INTO readings(session_id, ts, total, engaged, partial, disengaged,"
                  " class_pct, hands, phones) VALUES (?,?,?,?,?,?,?,?,?)",
                  (session_id, datetime.now().isoformat(timespec="seconds"), counts["total"],
                   counts["engaged"], counts["partial"], counts["disengaged"], pct,
                   counts["hands"], counts["phones"]))
        c.commit()


def log_event(session_id, etype, detail, pct=None):
    with closing(_conn()) as c:
        c.execute("INSERT INTO events(session_id, ts, etype, detail, pct) VALUES (?,?,?,?,?)",
                  (session_id, datetime.now().isoformat(timespec="seconds"), etype, detail, pct))
        c.commit()


def list_sessions():
    with closing(_conn()) as c:
        rows = c.execute("SELECT session_id FROM readings GROUP BY session_id "
                         "ORDER BY MAX(id) DESC").fetchall()
    return [r[0] for r in rows]


def session_summary():
    """One row per session: start, end, readings, average %, number of events."""
    with closing(_conn()) as c:
        return pd.read_sql_query(
            "SELECT r.session_id AS session_id, MIN(r.ts) AS start, MAX(r.ts) AS end,"
            " COUNT(*) AS n, AVG(r.class_pct) AS avg_pct,"
            " (SELECT COUNT(*) FROM events e WHERE e.session_id = r.session_id) AS n_events"
            " FROM readings r GROUP BY r.session_id ORDER BY MAX(r.id) DESC", c)


def read_session(session_id):
    with closing(_conn()) as c:
        return pd.read_sql_query("SELECT * FROM readings WHERE session_id=? ORDER BY ts",
                                 c, params=(session_id,))


def read_events(session_id):
    with closing(_conn()) as c:
        return pd.read_sql_query("SELECT ts, etype, detail, pct FROM events "
                                 "WHERE session_id=? ORDER BY ts, id", c, params=(session_id,))


def delete_session(session_id):
    with closing(_conn()) as c:
        c.execute("DELETE FROM readings WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM events WHERE session_id=?", (session_id,))
        c.commit()


def delete_older_than(days):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with closing(_conn()) as c:
        old = [r[0] for r in c.execute(
            "SELECT session_id FROM readings GROUP BY session_id HAVING MAX(ts) < ?", (cutoff,))]
        for sid in old:
            c.execute("DELETE FROM readings WHERE session_id=?", (sid,))
            c.execute("DELETE FROM events WHERE session_id=?", (sid,))
        c.commit()
    return len(old)


def delete_everything():
    with closing(_conn()) as c:
        c.execute("DELETE FROM readings")
        c.execute("DELETE FROM events")
        c.commit()
