# MIDAS

A local-first AI agent: CustomTkinter desktop UI, raw `ollama` SDK
orchestration (no LangChain), a modular `tools/` capability layer wired
through a single `TOOL_REGISTRY` in `main.py`, SQLite reminders, an
APScheduler morning briefing spoken via local TTS, and a small FastAPI
surface for phone-to-PC control.

## Prerequisites

- **Python 3.12+** (not 3.9 — `onnxruntime`, a `kokoro-onnx` dependency,
  dropped Python 3.9 wheels, and 3.9 itself is past end-of-life). If
  multiple Pythons are installed, use the `py -3.12` launcher explicitly
  for every command below.
- **Ollama** installed and running.

## 1. Install Ollama models

```
ollama pull qwen2.5:7b
```

`arch-router:1.5b` isn't in Ollama's standard library — pull it from
Hugging Face instead:

```
ollama pull hf.co/katanemo/Arch-Router-1.5B.gguf
```

If that fails with `realm host "huggingface.co" does not match original
host "hf.co"`, that's a known Ollama bug
([ollama/ollama#15661](https://github.com/ollama/ollama/issues/15661)).
Work around it with the fully-qualified host:

```
ollama pull huggingface.co/katanemo/Arch-Router-1.5B.gguf
```

Either way, the pulled tag won't be named `arch-router:1.5b` — alias it
to match `config.py` (cheap, local, no re-download):

```
ollama cp hf.co/katanemo/Arch-Router-1.5B.gguf:latest arch-router:1.5b
```

Also pull the vision model used for overnight notification-screenshot
extraction (see "Overnight notification summary" below):

```
ollama pull qwen2.5vl:3b
```

Confirm all three are present:

```
ollama list
```

## 2. Get the Kokoro TTS model files

MIDAS uses `kokoro-onnx` (ONNX Runtime, CPU by default — keeps TTS off
the VRAM your Ollama models are using). Download the int8-quantized
model (smallest/fastest on CPU) and its voice pack from the
[kokoro-onnx releases page](https://github.com/thewh1teagle/kokoro-onnx/releases)
and place them at:

```
MIDAS/models/kokoro-v1.0.int8.onnx
MIDAS/models/voices-v1.0.bin
```

These filenames must match `KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH` in
`config.py`. If the files are missing, `TTSEngine` logs a warning and
speech becomes a no-op — the rest of the app still runs.

## 3. Install Python dependencies

```
py -3.12 -m pip install -r requirements.txt
```

## 4. Configure

Edit `config.py` for your watchlist tickers, news feeds, briefing time,
`USER_NAME` (used for the "Good morning, {name}" greeting), and API port.
`API_HOST` is already set to `0.0.0.0` so the server is reachable over
Tailscale, not just loopback — the real access control is the token
below, not the bind address.

Set a real shared-secret token for the phone API — use `setx`, not
`set`, so it persists across terminal sessions instead of only the
current one:

```
setx MIDAS_API_TOKEN your-long-random-token
```

## 5. Run

```
py -3.12 main.py
```

This launches the desktop UI (main thread), starts the FastAPI server on
`0.0.0.0:8420` (background thread — reachable over Tailscale, see below),
and starts the APScheduler jobs
(morning briefing + reminder-due polling). Ollama must already be
running (`ollama serve`, or the Ollama desktop app) for chat, routing,
and the briefing's prose formatting to work.

## Phone-to-PC API

All routes require an `X-MIDAS-Token` header matching `MIDAS_API_TOKEN`.
Reachable over Tailscale — install Tailscale on the PC and phone (same
account/tailnet), then use the PC's `100.x.x.x` Tailscale IP in place of
`127.0.0.1` below. `API_HOST = "0.0.0.0"` is already set for this; the
token is what actually gates access, not the bind address.

- `POST /chat` `{"message": "...", "history": []}` → `{"response": "..."}`
- `POST /briefing/trigger` → runs the briefing now, returns the prose. No
  PC-side speech on purpose — this endpoint's only caller is remote (a
  phone automation doing its own TTS); the scheduled PC-native briefing
  in `scheduler/jobs.py` is a separate code path and still speaks locally.
- `GET /reminders` / `POST /reminders` `{"text": "...", "remind_at": "2026-07-14T08:00:00"}`
- `POST /notification/capture` — raw image bytes as the request body (not
  multipart/`UploadFile` — a phone Shortcut can send this as a plain
  "File" request body with nothing else to configure). Responds
  immediately with `{"status": "queued"}`; the actual vision-model
  extraction and storage happen afterward via FastAPI `BackgroundTasks`,
  since inference (plus a possible cold model load) can take longer than
  a phone's request timeout, and nothing on the phone side needs to wait
  for or see the result. See "Overnight notification summary" below.
- `GET /notification/latest` → `{"response": "..."}` with the most recent
  capture's extracted text (or `"No captures yet."`). Read-only — doesn't
  mark anything as summarized, so checking it (e.g. to confirm the first
  capture of the night worked) never causes that capture to be skipped
  from the real morning recap.

The practical no-code phone client is the iOS **Shortcuts** app: a
"Get Contents of URL" action with the token as a header, POST/GET as
appropriate, and (for `/chat`-shaped JSON responses) "Get Dictionary
Value" on key `response` to pull out the text before speaking it.

## Overnight notification summary

MIDAS can't read Instagram/WhatsApp DMs directly — no personal-account
API exists for that, and iOS has no equivalent of Android's Notification
Listener permission, so no app (including Shortcuts) can read another
app's notification content directly either. The workaround: iOS
Shortcuts has a **Take Screenshot** action, and a scheduled screenshot of
a locked, dark phone's lock screen genuinely captures whatever's
accumulated in the notification stack — confirmed by direct testing.

**Phone side** (built in Shortcuts, not code): a "Capture Notification"
shortcut (Take Screenshot → Save to Photo Album, wrapped as a standalone
shortcut and invoked via "Run Shortcut" from a Personal Automation, since
Take Screenshot can't run inline in an auto-firing automation) extended
with a POST to `/notification/capture`. Duplicate the Time of Day
automation across several points in the night (e.g. every 30-60 min) —
each capture just adds to the pile, nothing needs to be seen or heard
overnight. Optionally, wire just the *first* automation of the night to
also wait ~15-20s and then check `/notification/latest` + Speak Text, as
a "did this work tonight" confirmation before you fall asleep.

**PC side**: `core/orchestrator.py`'s `extract_notification_text()` runs
the screenshot through `config.VISION_MODEL` (`qwen2.5vl:3b`) with a
deliberately open-ended prompt ("describe every notification, one per
line" — a stricter single-line-format-plus-explicit-bailout prompt was
tried first and reliably produced false "no notifications visible"
results even on screenshots with obvious, legible content; the model
itself was fine, tested independently with a generic "describe this
image" prompt). Results land in the `notification_captures` SQLite table
(`tools/notifications.py`, same fetch-then-mark-consumed pattern as
`tools/reminders.py`'s reminder alerts). Each morning, `run_morning_briefing()`
pulls everything not yet summarized, and `BRIEFING_FORMAT_PROMPT` is
explicitly told these are cumulative overnight captures that may repeat
the same notification several times (it stays on the lock screen until
dismissed) and to consolidate duplicates into one mention rather than
reading the same message back multiple times.

## Adding a new tool

Tool invocation is a deterministic Python decision (arch-router picks a
route name, Python matches it to a registry entry) — the chat model is
never handed a list of tools and asked whether to call one. Small models
are unreliable at that judgment call in practice: during testing,
llama3.2:3b called the briefing tool on a plain "hey, how's it going"
and hallucinated a fake `wikipedia` tool call as plain text for "what's
the capital of France" — neither of those is possible once tool choice
lives in Python instead of model output.

To add a tool:

1. Write a deterministic function in `tools/<name>.py`, a paired schema
   dict (`parameters` doubles as the JSON schema for argument extraction;
   `description` becomes its arch-router route description), and — unless
   it should get an LLM prose pass like the briefing does — a
   `describe_<name>(result) -> str` formatter next to it (plain Python
   string formatting, no LLM call, no hallucination risk).
2. Import both in `main.py` and add one entry to `TOOL_REGISTRY` with a
   unique `"route"` key, plus either `"describe": describe_<name>` or
   `"speak_prose": True`.

No other file needs to change — `Orchestrator` builds its route list and
dispatch table from `TOOL_REGISTRY` alone. If the tool takes no required
arguments (like `list_reminders`), it runs with zero LLM calls at all.


## Future additions
<<<<<<< HEAD
1. ~~Personal agent personality~~ — done: TTS now speaks every chat reply, not just the briefing.
2. ~~Connect to iPhone~~ — done via Tailscale (see "Phone-to-PC API"). "Give access to computer" was explicitly discussed and shelved — needs careful scoping (allowlisted actions vs. open command execution) before any code gets written, not something to build casually. "Connect to Google account API" not started.
3. ~~Is not aware of current news~~ — done: chat now has real date/time injected into its system prompt, and news/market questions correctly route to live data instead of the model guessing or hallucinating.
4. Ship
=======
1. Personal agent personality
2. Connect to Iphone, maybe give access to computer, connect to google account api
3. Is not aware of current news
4. Ship 
>>>>>>> b4bdb17ef4b649cf4b58fdde635b7a1ed41829c3


## Additions:

7/20/2025

Added morning debrief feature.
Planning to build overnight messages summarization feature. To overcome the obstacle of instagram not having an inherent API to return user messages and apple not having a notification reader function, the plan is to take screenshots everytime a notification pops up and sends it to qwen OCR to analyze it,then puts it in a DB to summarize in the morning
<<<<<<< HEAD

7/26/2026

Built v1 of the overnight notification summarization feature described above — screenshot capture via a Shortcuts automation, `qwen2.5vl:3b` vision extraction, SQLite log, dedup in the morning briefing (see "Overnight notification summary"). 
=======
>>>>>>> b4bdb17ef4b649cf4b58fdde635b7a1ed41829c3
