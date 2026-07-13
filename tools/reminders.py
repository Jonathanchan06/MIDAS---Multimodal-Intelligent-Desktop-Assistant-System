"""Reminders / calendar-alert tool. Every function is a small, deterministic
unit against SQLite — no LLM calls, no shared state, safe to call from any
thread. Each schema's `parameters` doubles as the JSON schema the
orchestrator uses to grammar-constrain argument extraction, and its
`description` becomes that tool's arch-router route description.

The describe_* functions turn a tool's result into a user-facing
confirmation string via plain Python formatting — no LLM call, no
hallucination risk, instant.
"""

from __future__ import annotations

from datetime import datetime, timezone

import dateparser

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
                        "When the reminder is due, exactly as the user said "
                        "it (e.g. 'tomorrow at 5pm', 'in 2 hours', 'tonight "
                        "at 9pm', or an ISO 8601 datetime). Leave date/time "
                        "math to the caller — pass the phrase through as-is."
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


def _parse_datetime(value: str) -> datetime:
    """ISO 8601 first (fast path for already-normalized input), then
    dateparser for natural-language phrases like 'tomorrow at 5pm' — date
    math belongs in a deterministic library, not in a 3B model's head."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parsed = dateparser.parse(value, settings={"PREFER_DATES_FROM": "future"})
        if parsed is None:
            raise ValueError(f"could not understand date/time: {value!r}")
    return parsed.replace(microsecond=0)


def add_reminder(text: str, remind_at: str) -> dict:
    """Insert a reminder. Raises ValueError if remind_at can't be parsed
    as a datetime — callers (the orchestrator's tool-dispatch boundary)
    are expected to catch this and surface it back to the user."""
    parsed = _parse_datetime(remind_at)

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


def describe_add_reminder(result: dict) -> str:
    return f"Reminder set: \"{result['text']}\" for {result['remind_at']}."


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


def describe_list_reminders(results: list[dict]) -> str:
    if not results:
        return "You have no upcoming reminders."
    lines = [f"#{r['id']}: {r['text']} — due {r['remind_at']}" for r in results]
    return "Your reminders:\n" + "\n".join(lines)


def delete_reminder(reminder_id: int) -> dict:
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        conn.commit()
        return {"deleted": cursor.rowcount > 0, "id": reminder_id}
    finally:
        conn.close()


def describe_delete_reminder(result: dict) -> str:
    if result["deleted"]:
        return f"Deleted reminder #{result['id']}."
    return f"Couldn't find reminder #{result['id']}."


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
