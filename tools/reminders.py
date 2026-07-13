"""Reminders / calendar-alert tool. Every function is a small, deterministic
unit against SQLite — no LLM calls, no shared state, safe to call from any
thread. Schemas below are handed to Ollama's tool-calling API verbatim.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.db import get_connection

ADD_REMINDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_reminder",
        "description": (
            "Create a reminder or calendar alert for the user. Call this "
            "whenever the user asks to be reminded of something or wants "
            "to schedule an alert."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "What to remind the user about.",
                },
                "remind_at": {
                    "type": "string",
                    "description": (
                        "ISO 8601 local datetime the reminder is due, e.g. "
                        "2026-07-14T08:00:00."
                    ),
                },
            },
            "required": ["text", "remind_at"],
        },
    },
}

LIST_REMINDERS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_reminders",
        "description": "List the user's upcoming (not yet completed) reminders.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_done": {
                    "type": "boolean",
                    "description": "Include already-completed reminders too.",
                },
            },
            "required": [],
        },
    },
}

DELETE_REMINDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delete_reminder",
        "description": "Delete or cancel a reminder by its id.",
        "parameters": {
            "type": "object",
            "properties": {
                "reminder_id": {
                    "type": "integer",
                    "description": "The id of the reminder to delete.",
                },
            },
            "required": ["reminder_id"],
        },
    },
}


def add_reminder(text: str, remind_at: str) -> dict:
    """Insert a reminder. Raises ValueError if remind_at isn't a valid
    ISO 8601 datetime — callers (the orchestrator's tool-dispatch boundary)
    are expected to catch this and surface it back to the model."""
    parsed = datetime.fromisoformat(remind_at)

    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO reminders (text, remind_at, created_at, done) "
            "VALUES (?, ?, ?, 0)",
            (text, parsed.isoformat(), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "text": text,
            "remind_at": parsed.isoformat(),
            "done": False,
        }
    finally:
        conn.close()


def list_reminders(include_done: bool = False) -> list[dict]:
    conn = get_connection()
    try:
        if include_done:
            rows = conn.execute(
                "SELECT * FROM reminders ORDER BY remind_at ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE done = 0 ORDER BY remind_at ASC"
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_reminder(reminder_id: int) -> dict:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        conn.commit()
        return {"deleted": cursor.rowcount > 0, "id": reminder_id}
    finally:
        conn.close()


def get_due_reminders(now: str | None = None) -> list[dict]:
    """Not exposed as an LLM tool — used by the scheduler's polling job.
    Marks matching rows done so the same alert never fires twice."""
    reference = datetime.fromisoformat(now) if now else datetime.now()

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE done = 0 AND remind_at <= ?",
            (reference.isoformat(),),
        ).fetchall()
        due = [dict(row) for row in rows]
        if due:
            conn.executemany(
                "UPDATE reminders SET done = 1 WHERE id = ?",
                [(row["id"],) for row in due],
            )
            conn.commit()
        return due
    finally:
        conn.close()
