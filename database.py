import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT    NOT NULL,
    filename    TEXT    NOT NULL,
    person_count INTEGER NOT NULL DEFAULT 0,
    male_count  INTEGER NOT NULL DEFAULT 0,
    female_count INTEGER NOT NULL DEFAULT 0,
    car_count   INTEGER NOT NULL DEFAULT 0,
    object_counts TEXT,
    avg_confidence REAL  NOT NULL DEFAULT 0,
    density_level  TEXT,
    processing_time_ms INTEGER,
    alert       INTEGER NOT NULL DEFAULT 0,
    alert_threshold INTEGER,
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS video_frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analysis(id) ON DELETE CASCADE,
    second      REAL    NOT NULL,
    count       INTEGER NOT NULL DEFAULT 0,
    males       INTEGER NOT NULL DEFAULT 0,
    females     INTEGER NOT NULL DEFAULT 0,
    objects     TEXT,
    confidence  REAL    NOT NULL DEFAULT 0,
    processing_time_ms INTEGER
);

CREATE TABLE IF NOT EXISTS settings_store (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def save_analysis(data: dict) -> int:
    sql = """
        INSERT INTO analysis
            (type, filename, person_count, male_count, female_count, car_count,
             object_counts, avg_confidence, density_level, processing_time_ms,
             alert, alert_threshold, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    with get_conn() as conn:
        cur = conn.execute(sql, (
            data.get("type", "image"),
            data.get("filename", "unknown"),
            data.get("person_count", 0),
            data.get("male_count", 0),
            data.get("female_count", 0),
            data.get("car_count", 0),
            json.dumps(data.get("object_counts", {})),
            data.get("avg_confidence", 0),
            data.get("density_level", ""),
            data.get("processing_time_ms", 0),
            int(data.get("alert", False)),
            data.get("alert_threshold", 10),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ))
        return cur.lastrowid


def save_video_frames(analysis_id: int, frames: list):
    sql = """
        INSERT INTO video_frames
            (analysis_id, second, count, males, females, objects, confidence, processing_time_ms)
        VALUES (?,?,?,?,?,?,?,?)
    """
    with get_conn() as conn:
        conn.executemany(sql, [
            (
                analysis_id,
                f["second"],
                f["count"],
                f.get("males", 0),
                f.get("females", 0),
                json.dumps(f.get("objects", {})),
                f.get("confidence", 0),
                f.get("processing_time_ms", 0),
            )
            for f in frames
        ])


def get_history(limit: int = 20) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats() -> dict:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                        AS total_analyses,
                COALESCE(SUM(person_count), 0)  AS total_people_detected,
                COALESCE(MAX(person_count), 0)  AS max_people_in_single,
                COALESCE(AVG(avg_confidence), 0) AS overall_avg_confidence,
                COALESCE(SUM(alert), 0)         AS total_alerts,
                COALESCE(SUM(CASE WHEN type='image' THEN 1 ELSE 0 END), 0) AS image_count,
                COALESCE(SUM(CASE WHEN type='video' THEN 1 ELSE 0 END), 0) AS video_count,
                COALESCE(SUM(CASE WHEN type='webcam' THEN 1 ELSE 0 END), 0) AS webcam_count
            FROM analysis
        """).fetchone()
        return dict(row)


def clear_history():
    with get_conn() as conn:
        conn.execute("DELETE FROM analysis")


def get_setting(key: str, default: str = None) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM settings_store WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings_store(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value)
        )
