import sqlite3
from contextlib import closing
from datetime import datetime

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
            session_id TEXT, ts TEXT, event_type TEXT, detail TEXT, class_pct REAL, image TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS recordings(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, filename TEXT, start_ts TEXT, end_ts TEXT, fps REAL)""")
        try:                                   # older databases do not have the image column yet
            c.execute("ALTER TABLE events ADD COLUMN image TEXT")
        except sqlite3.OperationalError:
            pass
        c.commit()


def _now(ms=False):
    return datetime.now().isoformat(timespec="milliseconds" if ms else "seconds")


def log_reading(session_id, counts, pct):
    with closing(_conn()) as c:
        c.execute("INSERT INTO readings(session_id, ts, total, engaged, partial, disengaged,"
                  " class_pct, hands, phones) VALUES (?,?,?,?,?,?,?,?,?)",
                  (session_id, _now(), counts["total"], counts["engaged"], counts["partial"],
                   counts["disengaged"], pct, counts["hands"], counts["phones"]))
        c.commit()


def log_event(session_id, event_type, detail, class_pct=None, image=None):
    with closing(_conn()) as c:
        c.execute("INSERT INTO events(session_id, ts, event_type, detail, class_pct, image)"
                  " VALUES (?,?,?,?,?,?)", (session_id, _now(), event_type, detail, class_pct, image))
        c.commit()


def add_recording(session_id, filename, fps):
    with closing(_conn()) as c:
        cur = c.execute("INSERT INTO recordings(session_id, filename, start_ts, fps) VALUES (?,?,?,?)",
                        (session_id, filename, _now(ms=True), fps))
        c.commit()
        return cur.lastrowid


def end_recording(rec_id):
    with closing(_conn()) as c:
        c.execute("UPDATE recordings SET end_ts=? WHERE id=?", (_now(ms=True), rec_id))
        c.commit()


def read_recordings(session_id):
    with closing(_conn()) as c:
        return pd.read_sql_query("SELECT * FROM recordings WHERE session_id=? ORDER BY id",
                                 c, params=(session_id,))


def delete_recordings(session_id):
    """Remove recording rows and snapshot links; return (video files, image files) to delete."""
    with closing(_conn()) as c:
        vids = [r[0] for r in c.execute("SELECT filename FROM recordings WHERE session_id=?", (session_id,))]
        imgs = [r[0] for r in c.execute(
            "SELECT image FROM events WHERE session_id=? AND image IS NOT NULL", (session_id,))]
        c.execute("DELETE FROM recordings WHERE session_id=?", (session_id,))
        c.execute("UPDATE events SET image=NULL WHERE session_id=?", (session_id,))
        c.commit()
    return vids, imgs


def list_sessions():
    with closing(_conn()) as c:
        rows = c.execute("SELECT session_id FROM readings GROUP BY session_id "
                         "ORDER BY MAX(id) DESC").fetchall()
    return [r[0] for r in rows]


def read_session(session_id):
    with closing(_conn()) as c:
        return pd.read_sql_query("SELECT * FROM readings WHERE session_id=? ORDER BY ts, id",
                                 c, params=(session_id,))


def read_events(session_id):
    with closing(_conn()) as c:
        return pd.read_sql_query("SELECT * FROM events WHERE session_id=? ORDER BY ts, id",
                                 c, params=(session_id,))


def read_all():
    """All readings and events of every session (timestamps parsed)."""
    with closing(_conn()) as c:
        rd = pd.read_sql_query("SELECT * FROM readings ORDER BY ts, id", c)
        ev = pd.read_sql_query("SELECT * FROM events ORDER BY ts, id", c)
    rd["ts"], ev["ts"] = pd.to_datetime(rd["ts"]), pd.to_datetime(ev["ts"])
    return rd, ev


def clear_recordings():
    with closing(_conn()) as c:
        c.execute("DELETE FROM recordings")
        c.execute("UPDATE events SET image=NULL")
        c.commit()


def clear_all():
    with closing(_conn()) as c:
        for t in ("readings", "events", "recordings"):
            c.execute(f"DELETE FROM {t}")
        c.commit()
