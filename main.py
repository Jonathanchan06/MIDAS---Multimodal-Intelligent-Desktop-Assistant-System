"""MIDAS composition root.

Every capability lives as an isolated, deterministic function under
tools/. This file's only job is to wire those functions (plus their
Ollama-schema-shaped metadata) into one flat TOOL_REGISTRY dict and hand
it to the orchestrator — no if/else dispatch chains anywhere.

Each entry's "route" is the arch-router label Python matches on to invoke
that tool deterministically (see core/orchestrator.py) — the model never
chooses whether to call a tool. "describe" is a plain-Python formatter
for the tool's result; "speak_prose" opts into an LLM prose pass instead,
naming which prompt/style to format with (see Orchestrator._prose_formatters)
for a tool whose result is meant to be read aloud rather than shown as text.
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
from tools.wardrobe import (
    ADD_WARDROBE_ITEM_SCHEMA,
    DELETE_WARDROBE_ITEM_SCHEMA,
    LIST_WARDROBE_ITEMS_SCHEMA,
    SUGGEST_FIT_SCHEMA,
    UPDATE_WARDROBE_ITEM_SCHEMA,
    add_wardrobe_items,
    delete_wardrobe_items,
    describe_add_wardrobe_items,
    describe_delete_wardrobe_items,
    describe_list_wardrobe_items,
    describe_update_wardrobe_items,
    list_wardrobe_items,
    resolve_add_items,
    resolve_list_category,
    suggest_fit,
    update_wardrobe_items,
)
from ui.app import MidasApp

logging.basicConfig(level=logging.INFO)

TOOL_REGISTRY = {
    "run_morning_briefing": {
        "fn": run_morning_briefing,
        "schema": BRIEFING_SCHEMA,
        "route": "briefing",
        "speak_prose": "briefing",
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
    "suggest_fit": {
        "fn": suggest_fit,
        "schema": SUGGEST_FIT_SCHEMA,
        "route": "suggest_fit",
        "speak_prose": "outfit",
    },
    "add_wardrobe_item": {
        "fn": add_wardrobe_items,
        "schema": ADD_WARDROBE_ITEM_SCHEMA,
        "route": "add_wardrobe_item",
        "describe": describe_add_wardrobe_items,
        "post_extract": resolve_add_items,
    },
    "list_wardrobe_items": {
        "fn": list_wardrobe_items,
        "schema": LIST_WARDROBE_ITEMS_SCHEMA,
        "route": "list_wardrobe_items",
        "describe": describe_list_wardrobe_items,
        "post_extract": resolve_list_category,
    },
    "delete_wardrobe_item": {
        "fn": delete_wardrobe_items,
        "schema": DELETE_WARDROBE_ITEM_SCHEMA,
        "route": "delete_wardrobe_item",
        "describe": describe_delete_wardrobe_items,
    },
    "update_wardrobe_item": {
        "fn": update_wardrobe_items,
        "schema": UPDATE_WARDROBE_ITEM_SCHEMA,
        "route": "update_wardrobe_item",
        "describe": describe_update_wardrobe_items,
    },
}


def main() -> None:
    init_db()

    orchestrator = Orchestrator(tool_registry=TOOL_REGISTRY)
    tts_engine = TTSEngine()

    scheduler = start_scheduler(orchestrator, tts_engine)

    api_app = create_app(orchestrator)
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
