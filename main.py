"""MIDAS composition root.

Every capability lives as an isolated, deterministic function under
tools/. This file's only job is to wire those functions (plus their
Ollama-schema-shaped metadata) into one flat TOOL_REGISTRY dict and hand
it to the orchestrator — no if/else dispatch chains anywhere.

Each entry's "route" is the arch-router label Python matches on to invoke
that tool deterministically (see core/orchestrator.py) — the model never
chooses whether to call a tool. "describe" is a plain-Python formatter
for the tool's result; "speak_prose" opts into an LLM prose pass instead,
for the one tool (the briefing) that's meant to be read aloud.
"""

import logging
import threading

import uvicorn

import config
from core.db import init_db
from core.orchestrator import Orchestrator
from core.tts import TTSEngine
from scheduler.jobs import start_scheduler
from server.api import create_app
from tools.briefing import BRIEFING_SCHEMA, run_morning_briefing
from tools.reminders import (
    ADD_REMINDER_SCHEMA,
    DELETE_REMINDER_SCHEMA,
    LIST_REMINDERS_SCHEMA,
    add_reminder,
    delete_reminder,
    describe_add_reminder,
    describe_delete_reminder,
    describe_list_reminders,
    list_reminders,
)
from ui.app import MidasApp

logging.basicConfig(level=logging.INFO)

TOOL_REGISTRY = {
    "run_morning_briefing": {
        "fn": run_morning_briefing,
        "schema": BRIEFING_SCHEMA,
        "route": "briefing",
        "speak_prose": True,
    },
    "add_reminder": {
        "fn": add_reminder,
        "schema": ADD_REMINDER_SCHEMA,
        "route": "add_reminder",
        "describe": describe_add_reminder,
    },
    "list_reminders": {
        "fn": list_reminders,
        "schema": LIST_REMINDERS_SCHEMA,
        "route": "list_reminders",
        "describe": describe_list_reminders,
    },
    "delete_reminder": {
        "fn": delete_reminder,
        "schema": DELETE_REMINDER_SCHEMA,
        "route": "delete_reminder",
        "describe": describe_delete_reminder,
    },
}


def main() -> None:
    init_db()

    orchestrator = Orchestrator(tool_registry=TOOL_REGISTRY)
    tts_engine = TTSEngine()

    scheduler = start_scheduler(orchestrator, tts_engine)

    api_app = create_app(orchestrator, tts_engine)
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(api_app,),
        kwargs={"host": config.API_HOST, "port": config.API_PORT, "log_level": "warning"},
        daemon=True,
    )
    api_thread.start()

    app = MidasApp(orchestrator=orchestrator, tts_engine=tts_engine)
    try:
        app.mainloop()
    finally:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
