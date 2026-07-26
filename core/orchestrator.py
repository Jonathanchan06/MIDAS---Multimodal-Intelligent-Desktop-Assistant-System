"""Routing + dispatch over the raw ollama SDK (no LangChain / agent
framework).

Tool invocation is a deterministic Python decision, not something the
chat model chooses at generation time — small models are unreliable at
deciding *whether* to call a tool (observed in practice: spurious tool
calls on plain greetings, and hallucinated calls to tools that were never
declared). Instead:

  1. arch-router:1.5b classifies each message into one of a small set of
     routes built directly from TOOL_REGISTRY (plus a static "chat"
     fallback route) — a narrow, bounded classification task it's
     fine-tuned for.
  2. Python matches the route name to a registry entry directly. There is
     no step where a chat model is handed a list of tools and asked to
     decide whether/which one to call.
  3. If the matched tool has required arguments, one grammar-constrained
     extraction call (Ollama's JSON-schema `format`) pulls them out —
     forced-structure extraction, not "decide and call" tool-use.
  4. The chat model only ever runs with tools=None. It has nothing to
     hallucinate the shape of, because it never sees a tools list.

`_call_tool` remains the single enforced boundary between LLM-authored
arguments and real code execution: exceptions become a controlled result
instead of crashing the calling thread (UI worker, FastAPI handler, or
scheduler job).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable

import ollama

import config

logger = logging.getLogger(__name__)

STATIC_ROUTES = [
    {
        "name": "chat",
        "description": (
            "General conversation, greetings, opinions, small talk, general "
            "knowledge questions, or requests to write, debug, or explain "
            "code — anything not covered by any other route."
        ),
    },
]

# Verbatim prompt contract from Arch-Router-1.5B's model card. This is a
# narrow routing specialist, not a general instruction-follower — it was
# fine-tuned specifically against this <routes>/<conversation> XML
# structure plus a JSON {"route": ...} output, and ignores ad hoc
# instructions outside that contract.
ROUTER_TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""

ROUTER_FORMAT_PROMPT = """
Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags.  Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent.
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"}
"""

CHAT_SYSTEM_PROMPT = (
    "You are MIDAS, a concise local assistant running entirely on the "
    "user's own machine. Keep answers short and conversational, and "
    "always respond in English regardless of what language is implied "
    "elsewhere. Current date and time: {now}. You do NOT have live "
    "access to news, stock prices, or other current events in this "
    "mode — never invent or guess a specific headline, price, or event. "
    "If asked about any of those, say you don't have live access here "
    "and suggest asking for the briefing instead."
)

BRIEFING_FORMAT_PROMPT = (
<<<<<<< HEAD
    "Turn this raw briefing JSON into short spoken sentences, entirely in "
    "English: market moves first, then headlines, then a summary of "
    "overnight notifications if any are present. The notifications list "
    "may contain the same message captured more than once — screenshots "
    "were taken repeatedly through the night, so a notification that sat "
    "on the lock screen for hours can appear in several captures. "
    "Consolidate duplicates into a single mention per sender/topic, never "
    "repeat the same message. Plain prose, no markdown, no bullet points "
    "— this will be read aloud by a TTS engine."
=======
    "Turn this raw briefing JSON into 3-5 short spoken sentences, entirely "
    "in English: market moves first, then headlines. Plain prose, no "
    "markdown, no bullet points — this will be read aloud by a TTS engine."
>>>>>>> b4bdb17ef4b649cf4b58fdde635b7a1ed41829c3
)

EXTRACTION_SYSTEM_PROMPT = (
    "Current datetime: {now}. Extract the arguments for the user's "
    "request as JSON matching the given schema. Keep any free-text field "
    "a short paraphrase of the underlying request — do not restate a "
    "date/time that already belongs in its own field."
)

NOTIFICATION_EXTRACTION_PROMPT = (
    "Describe every notification visible in this image, one per line. "
    "For each one, mention which app it's from, who it's from, and what "
    "it says. If the image genuinely shows no notifications at all, say "
    "so plainly."
)


class Orchestrator:
    def __init__(self, tool_registry: dict):
        self.tool_registry = tool_registry
        self.client = ollama.Client(host=config.OLLAMA_HOST)

        self._route_to_tool: dict[str, tuple[str, dict]] = {}
        routes = list(STATIC_ROUTES)
        for name, entry in tool_registry.items():
            route_name = entry.get("route")
            if not route_name:
                continue
            routes.append(
                {"name": route_name, "description": entry["schema"]["function"]["description"]}
            )
            self._route_to_tool[route_name] = (name, entry)
        self._routes = routes

    def route(self, text: str, history: list[dict]) -> str:
        """Classifies into one of self._routes via Arch-Router's
        routes/conversation contract. Falls back to "chat" on failure or
        on any label that doesn't match a known route — never blocks a
        response."""
        conversation = json.dumps([*history, {"role": "user", "content": text}])
        prompt = (
            ROUTER_TASK_INSTRUCTION.format(
                routes=json.dumps(self._routes), conversation=conversation
            )
            + ROUTER_FORMAT_PROMPT
        )
        try:
            response = self.client.chat(
                model=config.ROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                keep_alive=config.ROUTER_KEEP_ALIVE,
                options={"num_ctx": config.ROUTER_NUM_CTX, "temperature": config.ROUTER_TEMPERATURE},
                stream=False,
                format="json",
            )
            label = json.loads(response.message.content).get("route")
            valid_names = {r["name"] for r in self._routes}
            return label if label in valid_names else "chat"
        except Exception:
            logger.exception("router call failed, defaulting to chat")
            return "chat"

    def _call_tool(self, name: str, arguments: dict) -> dict:
        entry = self.tool_registry.get(name)
        if entry is None:
            return {"ok": False, "error": f"unknown tool '{name}'"}
        try:
            return {"ok": True, "result": entry["fn"](**arguments)}
        except Exception as exc:
            logger.exception("tool '%s' raised", name)
            return {"ok": False, "error": str(exc)}

    def _extract_arguments(self, params_schema: dict, text: str, history: list[dict]) -> dict:
        messages = [
            {
                "role": "system",
                "content": EXTRACTION_SYSTEM_PROMPT.format(
                    now=datetime.now().isoformat(timespec="seconds")
                ),
            },
            *history,
            {"role": "user", "content": text},
        ]
        response = self.client.chat(
            model=config.ORCHESTRATOR_MODEL,
            messages=messages,
            format=params_schema,
            keep_alive=config.CHAT_KEEP_ALIVE,
            options={"num_ctx": config.CHAT_NUM_CTX, "temperature": 0.1},
            stream=False,
        )
        return json.loads(response.message.content)

    def _run_tool_route(self, tool_match: tuple[str, dict], text: str, history: list[dict]) -> str:
        name, entry = tool_match
        params = entry["schema"]["function"]["parameters"]

        arguments = {}
        if params.get("required"):
            try:
                arguments = self._extract_arguments(params, text, history)
            except Exception:
                logger.exception("argument extraction failed for '%s'", name)

        outcome = self._call_tool(name, arguments)
        if not outcome["ok"]:
            return f"Sorry, that didn't work: {outcome['error']}"
        if entry.get("speak_prose"):
            return self.format_briefing(outcome["result"])
        describe = entry.get("describe")
        if describe:
            return describe(outcome["result"])
        return json.dumps(outcome["result"], default=str)

    def _stream_plain(
        self,
        model: str,
        system_prompt: str,
        keep_alive: str,
        num_ctx: int,
        temperature: float,
        text: str,
        history: list[dict],
        on_token: Callable[[str], None],
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": text},
        ]
        full_text = []
        for chunk in self.client.chat(
            model=model,
            messages=messages,
            keep_alive=keep_alive,
            options={"num_ctx": num_ctx, "temperature": temperature},
            stream=True,
        ):
            content = chunk.message.content
            if content:
                on_token(content)
                full_text.append(content)
        return "".join(full_text)

    def stream_response(
        self,
        text: str,
        history: list[dict],
        on_token: Callable[[str], None],
        on_route: Callable[[str], None] | None = None,
    ) -> str:
        """Routes the message, then either streams tokens live (chat) or
        runs a deterministic tool dispatch and delivers its result as a
        single chunk to on_token. Returns the final full text either way.
        on_route (if given) is called once with the model name that ended
        up handling the request, so a UI can reflect it."""
        route = self.route(text, history)

        tool_match = self._route_to_tool.get(route)
        if tool_match:
            if on_route:
                on_route(config.ORCHESTRATOR_MODEL)
            full_text = self._run_tool_route(tool_match, text, history)
            on_token(full_text)
            return full_text

        if on_route:
            on_route(config.ORCHESTRATOR_MODEL)
        now = datetime.now().strftime("%A, %B %d, %Y, %H:%M")
        return self._stream_plain(
            config.ORCHESTRATOR_MODEL,
            CHAT_SYSTEM_PROMPT.format(now=now),
            config.CHAT_KEEP_ALIVE,
            config.CHAT_NUM_CTX,
            config.CHAT_TEMPERATURE,
            text,
            history,
            on_token,
        )

    def format_briefing(self, raw: dict) -> str:
        """One-shot call that turns a deterministic briefing payload into
        TTS-ready prose — the one place an LLM call is used purely for
        phrasing, per the original spec, rather than for a decision. The
        greeting is prepended in Python, not left to the model to
        remember — same reasoning as the reminder confirmations: a fixed,
        short prefix doesn't need to go through generation at all."""
        response = self.client.chat(
            model=config.ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": BRIEFING_FORMAT_PROMPT},
                {"role": "user", "content": json.dumps(raw, default=str)},
            ],
            keep_alive=config.CHAT_KEEP_ALIVE,
            options={"num_ctx": config.CHAT_NUM_CTX, "temperature": config.CHAT_TEMPERATURE},
<<<<<<< HEAD
            stream=False,
        )
        prose = response.message.content.strip()
        return f"Good morning, {config.USER_NAME}. {prose}"

    def extract_notification_text(self, image_bytes: bytes) -> str:
        """One-shot vision call: turns a lock-screen screenshot into a
        plain-text listing of visible notifications. Called directly by
        the /notification/capture endpoint — not part of the routed
        chat/tool-dispatch path, same as format_briefing."""
        response = self.client.chat(
            model=config.VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": NOTIFICATION_EXTRACTION_PROMPT,
                    "images": [image_bytes],
                }
            ],
            keep_alive=config.VISION_KEEP_ALIVE,
            options={"num_ctx": config.VISION_NUM_CTX},
=======
>>>>>>> b4bdb17ef4b649cf4b58fdde635b7a1ed41829c3
            stream=False,
        )
        prose = response.message.content.strip()
        return f"Good morning, {config.USER_NAME}. {prose}"
