"""FastAPI surface for phone-to-PC control. Bound to loopback by default
(config.API_HOST) so routing this to a Tailscale interface later is a
one-line change, not a redesign.

Every route is guarded by a shared-secret header (X-MIDAS-Token) — minimal
but appropriate auth for a solo local tool that's about to get a
tailnet-routable IP.
"""

from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import config
from tools.briefing import run_morning_briefing
from tools.reminders import add_reminder, list_reminders


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

    return app
