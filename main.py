"""MIDAS composition root.

Every capability lives as an isolated, deterministic function under
tools/. This file's only job is to wire those functions (plus their
Ollama tool schemas) into one flat TOOL_REGISTRY dict and hand it to the
orchestrator — no if/else dispatch chains anywhere.
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
    list_reminders,
)
from ui.app import MidasApp

logging.basicConfig(level=logging.INFO)

TOOL_REGISTRY = {
    "run_morning_briefing": {"fn": run_morning_briefing, "schema": BRIEFING_SCHEMA},
    "add_reminder": {"fn": add_reminder, "schema": ADD_REMINDER_SCHEMA},
    "list_reminders": {"fn": list_reminders, "schema": LIST_REMINDERS_SCHEMA},
    "delete_reminder": {"fn": delete_reminder, "schema": DELETE_REMINDER_SCHEMA},
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
