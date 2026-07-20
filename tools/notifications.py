"""Overnight notification-capture log. Deterministic SQLite storage only —
the actual vision-model extraction happens in core/orchestrator.py before
these functions are ever called; nothing here talks to Ollama.

Not exposed as arch-router tools: this pipeline is never something the
chat model decides to invoke. save_notification_capture() is called
directly by the /notification/capture endpoint (server/api.py), and
get_overnight_captures() is called directly by run_morning_briefing()
(tools/briefing.py) — both fixed, deterministic call sites, not
conversational actions.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.db import get_connection


def save_notification_capture(extracted_text: str) -> dict:
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO notification_captures (captured_at, extracted_text, summarized) "
            "VALUES (?, ?, 0)",
            (datetime.now(timezone.utc).isoformat(), extracted_text),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "extracted_text": extracted_text}
    finally:
        conn.close()


def get_latest_capture() -> dict | None:
    """Read-only — does not touch the summarized flag, so checking this
    (e.g. the first automation of the night confirming it worked) never
    causes a capture to be skipped from the real morning recap."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM notification_captures ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_overnight_captures() -> list[dict]:
    """Fetches not-yet-summarized captures and marks them summarized, so
    the same capture never appears in two mornings' briefings — same
    fetch-then-mark pattern as tools/reminders.py's get_due_reminders()."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM notification_captures WHERE summarized = 0 "
            "ORDER BY captured_at ASC"
        ).fetchall()
        captures = [dict(row) for row in rows]
        if captures:
            conn.executemany(
                "UPDATE notification_captures SET summarized = 1 WHERE id = ?",
                [(row["id"],) for row in captures],
            )
            conn.commit()
        return captures
    finally:
        conn.close()
