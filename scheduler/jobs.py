"""APScheduler jobs: the morning briefing cron trigger and a reminder-due
poll. Each job body is wrapped in try/except so one failure (a bad ticker,
a network blip) never kills the scheduler thread.
"""

import logging

from apscheduler.schedulers.background import BackgroundScheduler

import config
from tools.briefing import run_morning_briefing
from tools.reminders import get_due_reminders

logger = logging.getLogger(__name__)


def _run_morning_briefing_job(orchestrator, tts_engine) -> None:
    try:
        raw = run_morning_briefing()
        prose = orchestrator.format_briefing(raw)
        tts_engine.speak(prose)
    except Exception:
        logger.exception("morning briefing job failed")


def _check_due_reminders_job(tts_engine) -> None:
    try:
        due = get_due_reminders()
        for reminder in due:
            tts_engine.speak(f"Reminder: {reminder['text']}")
    except Exception:
        logger.exception("reminder poll job failed")


def start_scheduler(orchestrator, tts_engine) -> BackgroundScheduler:
    scheduler = BackgroundScheduler()

    hour, minute = (int(part) for part in config.BRIEFING_TIME.split(":"))
    scheduler.add_job(
        _run_morning_briefing_job,
        trigger="cron",
        hour=hour,
        minute=minute,
        args=[orchestrator, tts_engine],
        id="morning_briefing",
        replace_existing=True,
    )

    scheduler.add_job(
        _check_due_reminders_job,
        trigger="interval",
        minutes=config.REMINDER_POLL_INTERVAL_MINUTES,
        args=[tts_engine],
        id="reminder_poll",
        replace_existing=True,
    )

    scheduler.start()
    return scheduler
