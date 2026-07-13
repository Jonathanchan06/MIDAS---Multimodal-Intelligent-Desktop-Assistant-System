"""Routing + tool-calling orchestration over the raw ollama SDK (no
LangChain / agent framework).

Three small models share the 6GB VRAM budget in turn:
  - arch-router:1.5b classifies each message as "chat" or "code"
  - llama3.2:3b handles chat + all tool calling
  - qwen2.5-coder:3b handles code-shaped requests

`_call_tool` is the single enforced boundary between LLM-authored tool
arguments and real code execution: every call is wrapped so a bad
argument or a tool-internal exception becomes a result the model can see
instead of crashing the calling thread (UI worker, FastAPI handler, or
scheduler job).
"""

from __future__ import annotations

import json
import logging
from typing import Callable

import ollama

import config

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = (
    'Classify the user\'s message. Reply with exactly one word:\n'
    '"code" - writing, debugging, or explaining source code\n'
    '"chat" - anything else\n'
    "Reply with only that single word, nothing else."
)

CHAT_SYSTEM_PROMPT = (
    "You are MIDAS, a concise local assistant running entirely on the "
    "user's own machine. Keep answers short. Use the available tools "
    "when the user asks about the morning briefing, markets, news, or "
    "reminders instead of guessing."
)

CODER_SYSTEM_PROMPT = (
    "You are MIDAS in coding mode. Give correct, minimal code with brief "
    "explanations. No filler."
)

BRIEFING_FORMAT_PROMPT = (
    "Turn this raw briefing JSON into 3-5 short spoken sentences: market "
    "moves first, then headlines. Plain prose, no markdown, no bullet "
    "points — this will be read aloud by a TTS engine."
)


class Orchestrator:
    def __init__(self, tool_registry: dict):
        self.tool_registry = tool_registry
        self.client = ollama.Client(host=config.OLLAMA_HOST)

    def route(self, text: str) -> str:
        """One-word traffic classification. Defaults to "chat" on any
        failure or unparseable output — never blocks a response."""
        try:
            response = self.client.chat(
                model=config.ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                keep_alive=config.ROUTER_KEEP_ALIVE,
                options={"num_ctx": config.ROUTER_NUM_CTX},
                stream=False,
            )
            label = response.message.content.strip().lower()
            return "code" if "code" in label else "chat"
        except Exception:
            logger.exception("router call failed, defaulting to chat")
            return "chat"

    def _call_tool(self, name: str, arguments: dict) -> str:
        entry = self.tool_registry.get(name)
        if entry is None:
            return f"error: unknown tool '{name}'"
        try:
            result = entry["fn"](**arguments)
            return json.dumps(result, default=str)
        except Exception as exc:
            logger.exception("tool '%s' raised", name)
            return f"error: tool '{name}' failed: {exc}"

    def stream_response(
        self,
        text: str,
        history: list[dict],
        on_token: Callable[[str], None],
        on_route: Callable[[str], None] | None = None,
    ) -> str:
        """Routes the message, streams tokens to on_token as they arrive,
        transparently executing any tool calls the model requests, and
        returns the final full text. on_route (if given) is called once
        with the model name that ended up handling the request, so a UI
        can reflect it without spending a second routing call."""
        route = self.route(text)
        if route == "code":
            model = config.CODER_MODEL
            system_prompt = CODER_SYSTEM_PROMPT
            keep_alive = config.CODER_KEEP_ALIVE
            num_ctx = config.CODER_NUM_CTX
            tools = None
        else:
            model = config.ORCHESTRATOR_MODEL
            system_prompt = CHAT_SYSTEM_PROMPT
            keep_alive = config.CHAT_KEEP_ALIVE
            num_ctx = config.CHAT_NUM_CTX
            tools = [entry["schema"] for entry in self.tool_registry.values()]

        if on_route:
            on_route(model)

        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": text},
        ]

        tool_calls, full_text = self._consume_stream(
            model, messages, keep_alive, num_ctx, tools, on_token
        )

        if not tool_calls:
            return "".join(full_text)

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": call.function.name,
                            "arguments": dict(call.function.arguments),
                        }
                    }
                    for call in tool_calls
                ],
            }
        )
        for call in tool_calls:
            result = self._call_tool(call.function.name, dict(call.function.arguments))
            messages.append(
                {"role": "tool", "name": call.function.name, "content": result}
            )

        _, full_text = self._consume_stream(
            model, messages, keep_alive, num_ctx, None, on_token
        )
        return "".join(full_text)

    def _consume_stream(self, model, messages, keep_alive, num_ctx, tools, on_token):
        chat_kwargs = dict(
            model=model,
            messages=messages,
            keep_alive=keep_alive,
            options={"num_ctx": num_ctx},
            stream=True,
        )
        if tools:
            chat_kwargs["tools"] = tools

        tool_calls = []
        full_text = []
        for chunk in self.client.chat(**chat_kwargs):
            message = chunk.message
            if getattr(message, "tool_calls", None):
                tool_calls.extend(message.tool_calls)
            if message.content:
                on_token(message.content)
                full_text.append(message.content)
        return tool_calls, full_text

    def format_briefing(self, raw: dict) -> str:
        """One-shot, no-tools call that turns a deterministic briefing
        payload into TTS-ready prose."""
        response = self.client.chat(
            model=config.ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": BRIEFING_FORMAT_PROMPT},
                {"role": "user", "content": json.dumps(raw, default=str)},
            ],
            keep_alive=config.CHAT_KEEP_ALIVE,
            options={"num_ctx": config.CHAT_NUM_CTX},
            stream=False,
        )
        return response.message.content.strip()
