"""SQLite schema init and connection helper.

Every caller opens and closes its own connection (see tools/reminders.py) —
there is no shared connection object to guard across threads. This keeps
each tool function deterministic and safe to call from the UI thread, the
FastAPI worker threads, and the APScheduler thread without any locking.
"""

import sqlite3

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    remind_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS notification_captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    extracted_text TEXT NOT NULL,
    summarized INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS wardrobe_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    name TEXT NOT NULL,
    color TEXT,
    warmth_level INTEGER,
    formality INTEGER,
    last_worn_at TEXT,
    tags TEXT,
    layer TEXT,
    embedding BLOB,
    created_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_wardrobe_layer_column(conn: sqlite3.Connection) -> None:
    """Migration for databases created before the 'layer' column existed.
    CREATE TABLE IF NOT EXISTS is a no-op against a table that already
    exists, so a column added to _SCHEMA later doesn't retroactively
    apply to it -- this adds it directly if missing. Safe to call
    against a brand-new table too (it'll already have the column)."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(wardrobe_items)")}
    if "layer" not in columns:
        conn.execute("ALTER TABLE wardrobe_items ADD COLUMN layer TEXT")


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA)
        _ensure_wardrobe_layer_column(conn)
        conn.commit()
    finally:
        conn.close()
