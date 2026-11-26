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

import re
from datetime import datetime, timedelta, timezone

import dateparser

from core.db import get_connection

# Used when a day is given with no time ("remind me to pay rent tomorrow").
DEFAULT_REMINDER_TIME = "09:00"
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PART_OF_DAY = {
    "morning": "09:00", "noon": "12:00", "midday": "12:00", "afternoon": "14:00",
    "evening": "18:00", "tonight": "21:00", "night": "21:00", "midnight": "00:00",
}

ADD_REMINDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_reminder",
        "description": (
            "Create a NEW reminder or calendar alert for the user. Call "
            "this when the user asks to be reminded of something not yet "
            "saved — e.g. 'remind me to call mom at 5pm', 'set a reminder "
            "for the meeting tomorrow'. Do NOT call this when the user "
            "asks to see, check, or hear reminders that already exist "
            "(e.g. 'remind me of my tasks', 'what are my reminders') — "
            "that's list_reminders."
        ),
        # Structured slots, not a datetime: the model only copies out which
        # day / what clock time / how many minutes the user said, and
        # resolve_when does the calendar math. Asked for an ISO datetime,
        # qwen3:8b and qwen2.5:7b both put "next Tuesday" on the wrong
        # day; asked for the raw phrase, dateparser can't read "tonight",
        # "next Tuesday", or "Saturday morning".
        # `day` is an enum and in_minutes/date are optional on purpose: as
        # free text / required, both models filled them with junk (day
        # "d+1", "log"; in_minutes 1440 for "tomorrow at 8am").
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "What to remind the user about, without the date or "
                        "time (e.g. 'call mom')."
                    ),
                },
                "day": {
                    "type": "string",
                    "enum": ["", "today", "tomorrow", *_WEEKDAYS, "date"],
                    "description": (
                        "Which day the user said: 'today' (also for "
                        "'tonight'), 'tomorrow', a weekday (also for 'next "
                        "Tuesday'), or 'date' for a calendar date (then fill "
                        "'date'). Empty string if no day was said or the "
                        "time is relative ('in 2 hours')."
                    ),
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Only when day is 'date': the calendar date as MM-DD "
                        "(e.g. '12-03' for December 3). Omit otherwise."
                    ),
                },
                "time": {
                    "type": "string",
                    "description": (
                        "Clock time as 24-hour HH:MM (e.g. '15:00' for 3pm, "
                        "'12:00' for noon). If only a part of the day was "
                        "said: morning '09:00', afternoon '14:00', evening "
                        "'18:00', tonight '21:00'. Empty string if no time "
                        "was said or the time is relative."
                    ),
                },
                "in_minutes": {
                    "type": "integer",
                    "description": (
                        "Only when the user gave a relative time: minutes "
                        "from now (30 for 'in 30 minutes', 120 for 'in 2 "
                        "hours'). Omit otherwise."
                    ),
                },
            },
            "required": ["text", "day", "time"],
        },
    },
}

LIST_REMINDERS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_reminders",
        "description": (
            "List the user's upcoming (not yet completed) reminders. Call "
            "this when the user asks to see, check, or hear reminders "
            "they already have — e.g. 'what are my reminders', 'remind me "
            "of my existing tasks', 'what do I have coming up', 'show my "
            "reminders'. Do NOT call this to create a new reminder — "
            "that's add_reminder."
        ),
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
        "name": "delete_reminders",
        "description": (
            "Delete or cancel one or more of the user's existing reminders "
            "— e.g. 'cancel my dentist reminder', 'delete reminder #3', "
            "'never mind that reminder about the meeting'. Do NOT call "
            "this to create or list reminders."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reminders": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One entry per reminder to delete: the words naming "
                        "it (e.g. 'dentist' for 'cancel my dentist "
                        "reminder', 'call mom' for 'delete the reminder to "
                        "call mom'), or just the number if the user gave "
                        "one (e.g. '3' for 'delete reminder #3'). Never "
                        "make up a number."
                    ),
                },
            },
            "required": ["reminders"],
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


def _parse_clock(value: str) -> tuple[int, int] | None:
    """'15:00', '3pm', '3:30 pm', 'noon', 'evening' -> (hour, minute).
    The schema asks for HH:MM; the rest is tolerance, not the contract."""
    value = value.strip().lower().replace(".", "")
    if value in _PART_OF_DAY:
        value = _PART_OF_DAY[value]
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(?::\d{2})?\s*(am|pm)?", value)
    if not match:
        return None
    hour, minute, meridiem = int(match[1]), int(match[2] or 0), match[3]
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def resolve_when(day: str = "", time: str = "", in_minutes: int = 0, date: str = "",
                 now: datetime | None = None) -> datetime:
    """Turn add_reminder's extracted slots into a datetime — the calendar
    math the extraction model is deliberately not asked to do. Raises
    ValueError when there's nothing usable or the moment has passed, so
    the reply says so instead of setting a reminder at a guessed time."""
    now = (now or datetime.now()).replace(microsecond=0)
    if in_minutes and in_minutes > 0:
        return now + timedelta(minutes=in_minutes)

    day = re.sub(r"^(next|this|on)\s+", "", (day or "").strip().lower())
    # A date with no day means the date. A weekday/today/tomorrow still wins
    # over a filled-in date: models also fill `date` with their own (wrong)
    # calendar math, which is exactly what the day slot avoids.
    if day == "" and (date or "").strip():
        day = "date"
    if day == "date":
        day = (date or "").strip()
        if not day:
            raise ValueError("I didn't catch which date")
        if re.fullmatch(r"\d{1,2}-\d{1,2}", day):
            # MM-DD: dateparser would read "12-03" day-first in some locales.
            month, dom = map(int, day.split("-"))
            day = f"{now.year}-{month:02d}-{dom:02d}"
    clock = _parse_clock(time) if (time or "").strip() else None
    if (time or "").strip() and clock is None:
        raise ValueError(f"could not understand the time {time!r}")
    if day == "tonight" and clock is None:
        clock = _parse_clock("tonight")

    if not day:
        if clock is None:
            raise ValueError("I didn't catch when to remind you")
        at = now.replace(hour=clock[0], minute=clock[1], second=0)
        return at if at > now else at + timedelta(days=1)

    if day in ("today", "tonight"):
        date = now.date()
    elif day == "tomorrow":
        date = (now + timedelta(days=1)).date()
    elif day in _WEEKDAYS:
        ahead = (_WEEKDAYS.index(day) - now.weekday()) % 7
        date = (now + timedelta(days=ahead)).date()
        # Today's own weekday means today only if that time is still ahead.
        if ahead == 0:
            hour, minute = clock or (int(x) for x in DEFAULT_REMINDER_TIME.split(":"))
            if now.replace(hour=hour, minute=minute, second=0) <= now:
                date = (now + timedelta(days=7)).date()
    else:
        parsed = dateparser.parse(day, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now})
        if parsed is None:
            raise ValueError(f"could not understand the day {day!r}")
        date = parsed.date()
        # "12-03" read as this year's date can already be past — that
        # means next year's, the way a person saying "December 3" would.
        if date < now.date():
            date = date.replace(year=date.year + 1)

    hour, minute = clock or (int(x) for x in DEFAULT_REMINDER_TIME.split(":"))
    at = datetime.combine(date, datetime.min.time()).replace(hour=hour, minute=minute)
    if at <= now:
        raise ValueError(f"{at:%A %H:%M} has already passed")
    return at


# "in 30 minutes", "in an hour", "in half an hour", "in 2 days".
_RELATIVE_TIME = re.compile(
    r"\bin\s+(?:[\w-]+\s+){1,3}?(?:mins?|minutes?|hrs?|hours?|days?|weeks?)\b", re.IGNORECASE
)


def resolve_reminder_slots(text: str, arguments: dict) -> dict:
    """post_extract hook for add_reminder: keep in_minutes only if the
    user actually gave a relative time. qwen3 fills it alongside day/time
    anyway (observed: in_minutes=1440 for "tomorrow at 8am", 126000 for
    "tonight at 9pm"), and since in_minutes wins in resolve_when that
    turned correct day/time slots into a wrong reminder."""
    if arguments.get("in_minutes") and not _RELATIVE_TIME.search(text):
        return {**arguments, "in_minutes": 0}
    return arguments


def add_reminder_from_slots(text: str, day: str = "", time: str = "", in_minutes: int = 0,
                            date: str = "") -> dict:
    """Registered (LLM-path) tool entry point: resolve the extracted slots,
    then insert. The API's POST /reminders calls add_reminder directly
    with a phrase or ISO datetime instead."""
    return add_reminder(text, resolve_when(day, time, in_minutes, date).isoformat())


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


def speak_list_reminders(results: list[dict]) -> str:
    """Spoken counterpart to describe_list_reminders: ids and raw ISO
    timestamps read aloud are noise, so name the next couple and point
    at the screen for the rest."""
    if not results:
        return "You have no upcoming reminders."
    upcoming = "; ".join(r["text"] for r in results[:2])
    if len(results) <= 2:
        return f"You have {len(results)} reminder{'s' if len(results) > 1 else ''}: {upcoming}."
    return f"You have {len(results)} reminders. Next up: {upcoming}. The rest are on screen."


# Words that never identify which reminder is meant.
_REF_STOPWORDS = {
    "my", "the", "a", "an", "that", "this", "one", "ones", "to", "about",
    "for", "of", "on", "at", "reminder", "reminders", "remind", "me", "task",
    "tasks", "alert", "alerts", "number", "no", "id", "with", "and",
    # qwen2.5 writes the action into the reference ("cancel meeting
    # reminder", "Cancel reminder with ID #6") — verbs never name one.
    "cancel", "delete", "remove", "scratch", "forget", "clear", "drop",
}
# Clock times in a reference ("my 6am airport reminder") describe when,
# not what — the reminder's text doesn't contain them.
_TIME_TOKEN = re.compile(r"^\d{1,2}(:\d{2})?(am|pm)$|^\d{1,2}:\d{2}$")


def _ref_tokens(text: str | None) -> set[str]:
    """Lowercased words minus filler and clock times, with a crude plural
    strip — applied to both sides of a comparison, so it only has to be
    consistent (same idea as tools/wardrobe.py's _ref_tokens)."""
    words = re.findall(r"[a-z0-9:]+", (text or "").lower().replace("'s", ""))
    out = set()
    for word in words:
        if word in _REF_STOPWORDS or _TIME_TOKEN.match(word):
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


def _match_reminder(ref: str, user_text: str) -> tuple[dict | None, str | None]:
    """Resolve one reference ('dentist', 'call mom', '3') to exactly one
    reminder. Returns (row, None) or (None, reason); the caller must not
    delete anything unresolved.

    This replaced asking the extraction model for a reminder_id: it never
    sees the table, so it invented one (observed: reminder_id 0 for
    "cancel my reminder to call mom", 2026112518043700 for another) and
    the wrong row — or none — got deleted."""
    wanted = _ref_tokens(ref)
    if not wanted:
        return None, "I didn't catch which reminder you meant"
    if wanted <= {"all", "every", "everything", "everyth"}:
        return None, "I only delete reminders you name, not all of them at once"

    if len(wanted) == 1 and next(iter(wanted)).isdigit():
        number = next(iter(wanted))
        # An id only counts if the user actually said that number.
        if not re.search(rf"(?<!\d){number}(?!\d)", user_text):
            return None, "I didn't catch which reminder you meant"
        row = next((r for r in list_reminders(include_done=True) if r["id"] == int(number)), None)
        return (row, None) if row else (None, f"there's no reminder #{number}")

    def covers(row: dict) -> bool:
        # Every word the user used must appear in the reminder, allowing
        # a word to sit inside a longer one ('meds' in 'medication').
        have = _ref_tokens(row["text"])
        return all(any(w == h or (len(w) >= 3 and h.startswith(w)) for h in have) for w in wanted)

    candidates = [row for row in list_reminders() if covers(row)]
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, f"no upcoming reminder matches '{ref}'"
    options = ", ".join(f"#{row['id']} {row['text']}" for row in candidates[:5])
    return None, f"'{ref}' could be {options} — which one?"


def resolve_reminder_refs(text: str, arguments: dict) -> dict:
    """post_extract hook for delete_reminders: turn each reference into a
    real reminder id. Deleting is irreversible, so an ambiguous or
    unmatched reference deletes nothing and is reported back instead."""
    refs = [str(r or "") for r in arguments.get("reminders") or []]
    # Drop blank/filler entries the model sometimes adds, unless that
    # leaves nothing — then keep one so the reply says what went wrong.
    refs = [r for r in refs if _ref_tokens(r)] or refs[:1]
    lowered = text.lower()
    reminder_ids, unresolved = [], []
    for ref in refs:
        row, reason = _match_reminder(ref, lowered)
        if row is None:
            unresolved.append({"reminder": ref, "reason": reason})
        elif row["id"] not in reminder_ids:
            reminder_ids.append(row["id"])
    if not refs:
        unresolved.append({"reminder": "", "reason": "I didn't catch which reminder you meant"})
    return {"reminder_ids": reminder_ids, "unresolved": unresolved}


def delete_reminders(reminder_ids: list[int], unresolved: list[dict] | None = None) -> dict:
    """Registered tool entry point. Expects ids already resolved by
    resolve_reminder_refs; with no ids it deletes nothing."""
    conn = get_connection()
    try:
        deleted = []
        for reminder_id in reminder_ids or []:
            row = conn.execute("SELECT id, text FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
            if row is None:
                continue
            conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
            deleted.append(dict(row))
        conn.commit()
        return {"deleted": deleted, "unresolved": unresolved or []}
    finally:
        conn.close()


def describe_delete_reminders(result: dict) -> str:
    parts = []
    if result["deleted"]:
        names = ", ".join(f"\"{r['text']}\"" for r in result["deleted"])
        noun = "reminder" if len(result["deleted"]) == 1 else "reminders"
        parts.append(f"Deleted {noun}: {names}.")
    if result["unresolved"]:
        reasons = "; ".join(entry["reason"] for entry in result["unresolved"])
        parts.append(f"Didn't delete anything for that: {reasons}" + ("" if reasons.endswith("?") else "."))
    return " ".join(parts) or "Nothing was deleted."


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
