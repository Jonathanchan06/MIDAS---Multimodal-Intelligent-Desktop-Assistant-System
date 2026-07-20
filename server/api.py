"""FastAPI surface for phone-to-PC control. Bound to loopback by default
(config.API_HOST) so routing this to a Tailscale interface later is a
one-line change, not a redesign.

Every route is guarded by a shared-secret header (X-MIDAS-Token) — minimal
but appropriate auth for a solo local tool that's about to get a
tailnet-routable IP.
"""

import logging
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import config
from tools.briefing import run_morning_briefing
from tools.notifications import get_latest_capture, save_notification_capture
from tools.reminders import add_reminder, list_reminders

logger = logging.getLogger(__name__)


def _require_token(x_midas_token: Optional[str] = Header(default=None)) -> None:
    if x_midas_token != config.API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid or missing X-MIDAS-Token")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str


class ReminderRequest(BaseModel):
    text: str
    remind_at: str


class NotificationCaptureResponse(BaseModel):
    status: str


def _process_notification_capture(orchestrator, image_bytes: bytes) -> None:
    """Runs after the HTTP response has already been sent (see
    BackgroundTasks below) — vision inference on a full-res image, plus a
    possible cold model load, can genuinely take longer than a phone's
    request timeout. Nobody's watching this response overnight anyway
    (see the /notification/capture route), so there's no reason to make
    the phone wait for it. Exceptions are caught and logged here since
    there's no client left to report them to."""
    try:
        extracted_text = orchestrator.extract_notification_text(image_bytes)
        save_notification_capture(extracted_text)
    except Exception:
        logger.exception("background notification capture failed")


def create_app(orchestrator) -> FastAPI:
    app = FastAPI(title="MIDAS", dependencies=[Depends(_require_token)])

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        tokens: list[str] = []

        def collect(token: str) -> None:
            tokens.append(token)

        full_text = await run_in_threadpool(
            orchestrator.stream_response, request.message, request.history, collect
        )
        return ChatResponse(response=full_text or "".join(tokens))

    @app.post("/briefing/trigger", response_model=ChatResponse)
    async def trigger_briefing():
        # No local (PC-side) speech here on purpose — this endpoint's only
        # caller is remote (e.g. a phone automation), which does its own
        # TTS. The scheduled PC-native briefing in scheduler/jobs.py is a
        # separate code path and still speaks locally as before.
        raw = await run_in_threadpool(run_morning_briefing)
        prose = await run_in_threadpool(orchestrator.format_briefing, raw)
        return ChatResponse(response=prose)

    @app.get("/reminders")
    async def get_reminders(include_done: bool = False):
        return await run_in_threadpool(list_reminders, include_done)

    @app.post("/reminders")
    async def create_reminder(request: ReminderRequest):
        return await run_in_threadpool(add_reminder, request.text, request.remind_at)

    @app.post("/notification/capture", response_model=NotificationCaptureResponse)
    async def capture_notification(request: Request, background_tasks: BackgroundTasks):
        # Raw bytes, not UploadFile/multipart — the phone sends the
        # screenshot as a plain "File" request body (Shortcuts' simplest
        # upload option), so there's no multipart form to parse.
        image_bytes = await request.body()
        # Respond immediately — vision inference (plus a possible cold
        # model load) can take longer than a phone's request timeout, and
        # nothing on the phone side needs to wait for or see the result.
        background_tasks.add_task(_process_notification_capture, orchestrator, image_bytes)
        return NotificationCaptureResponse(status="queued")

    @app.get("/notification/latest", response_model=ChatResponse)
    async def latest_notification():
        # Read-only — doesn't touch the "summarized" flag, so checking
        # this (e.g. the first automation of the night confirming it
        # worked) never causes a capture to be skipped from the real
        # morning recap in run_morning_briefing(). Same {"response": ...}
        # shape as /chat and /briefing/trigger so the phone-side Shortcut
        # can reuse the exact Get Dictionary Value setup already built.
        capture = await run_in_threadpool(get_latest_capture)
        if capture is None:
            return ChatResponse(response="No captures yet.")
        return ChatResponse(response=capture["extracted_text"])

    return app
