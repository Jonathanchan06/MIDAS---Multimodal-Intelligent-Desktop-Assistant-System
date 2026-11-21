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
import sys
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

LOG_FILE = config.DATA_DIR / "midas.log"
# Raw stdout/stderr when there's no console — library prints, Tk callback
# tracebacks. Separate from LOG_FILE because Windows won't let the
# rotating handler rename a file someone else holds open.
CONSOLE_LOG_FILE = config.DATA_DIR / "midas.console.log"


def _configure_logging() -> None:
    """Always log to data/midas.log (rotating), plus the console when one
    exists. Under pythonw.exe — how MIDAS runs in the background, e.g.
    from a startup shortcut — sys.stdout/stderr are None: anything that
    writes to them would crash, notably uvicorn's default log config,
    which kills the API thread at startup. Point them at a file instead."""
    from logging.handlers import RotatingFileHandler

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None or sys.stderr is None:
        stream = open(CONSOLE_LOG_FILE, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
        has_console = False
    else:
        has_console = True

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handlers: list[logging.Handler] = [file_handler]
    if has_console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)

    # Background threads (API, scheduler, chat workers) would otherwise
    # report crashes only to stderr — get them into midas.log too.
    def log_thread_exception(args):
        logging.getLogger("midas").error(
            "uncaught exception in thread %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = log_thread_exception

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


_SINGLE_INSTANCE_MUTEX = "Local\\MIDAS-single-instance"
_ERROR_ALREADY_EXISTS = 183
_instance_mutex = None  # held for the process lifetime; released by the OS on exit


def _claim_single_instance() -> bool:
    """Closing the window now only hides it (the floating orb keeps MIDAS
    running), so relaunching is easy to do by accident — and a second
    copy would silently fail to bind the API port and fire the morning
    briefing twice. A named Windows mutex lets only the first one run."""
    global _instance_mutex
    if sys.platform != "win32":
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    _instance_mutex = kernel32.CreateMutexW(None, False, _SINGLE_INSTANCE_MUTEX)
    if ctypes.get_last_error() != _ERROR_ALREADY_EXISTS:
        return True
    # Visible even under pythonw.exe, where there's no console to print to.
    ctypes.windll.user32.MessageBoxW(
        None,
        "MIDAS is already running — click the floating orb to open it.",
        "MIDAS",
        0x40,  # MB_ICONINFORMATION
    )
    return False


def main() -> None:
    _configure_logging()
    if not _claim_single_instance():
        return
    init_db()

    orchestrator = Orchestrator(tool_registry=TOOL_REGISTRY)
    tts_engine = TTSEngine()

    scheduler = start_scheduler(orchestrator, tts_engine)

    api_app = create_app(orchestrator)
    api_thread = threading.Thread(
        target=uvicorn.run,
        args=(api_app,),
        kwargs={
            "host": config.API_HOST,
            "port": config.API_PORT,
            "log_level": "warning",
            # Use our logging setup (see _configure_logging) instead of
            # uvicorn's own, which needs a real stderr and crashes under
            # pythonw.exe.
            "log_config": None,
        },
        daemon=True,
    )
    api_thread.start()

    app = MidasApp(orchestrator=orchestrator, tts_engine=tts_engine)
    # Start-at-login launches pass this so signing in shows just the
    # floating orb, not the full window.
    if "--background" in sys.argv[1:]:
        app.hide_window()
    try:
        app.mainloop()
    finally:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
