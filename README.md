# MIDAS

A local-first AI agent: CustomTkinter desktop UI, raw `ollama` SDK
orchestration (no LangChain), a modular `tools/` capability layer wired
through a single `TOOL_REGISTRY` in `main.py`, SQLite reminders, an
APScheduler morning briefing spoken via local TTS, and a small FastAPI
surface for phone-to-PC control.

## 1. Install Ollama models

```
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:3b
```

`arch-router:1.5b` (traffic router) may not be in the standard Ollama
library yet. If `ollama pull arch-router:1.5b` fails, import it from the
GGUF on Hugging Face instead:

```
ollama pull hf.co/katanemo/Arch-Router-1.5B-GGUF
```

...and update `ROUTER_MODEL` in `config.py` to whatever tag that creates.

## 2. Get the Kokoro TTS model files

MIDAS uses `kokoro-onnx` (ONNX Runtime, CPU by default — keeps TTS off
the VRAM your Ollama models are using). Download `kokoro-v0_19.onnx` and
`voices.bin` from the [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
and place them at:

```
MIDAS/models/kokoro-v0_19.onnx
MIDAS/models/voices.bin
```

If these files are missing, `TTSEngine` logs a warning and speech becomes
a no-op — the rest of the app still runs.

## 3. Install Python dependencies

```
pip install -r requirements.txt
```

## 4. Configure

Edit `config.py` for your watchlist tickers, news feeds, briefing time,
and API port. Set a real shared-secret token for the phone API before
exposing it beyond loopback:

```
set MIDAS_API_TOKEN=your-long-random-token
```

## 5. Run

```
python main.py
```

This launches the desktop UI (main thread), starts the FastAPI server on
`127.0.0.1:8420` (background thread), and starts the APScheduler jobs
(morning briefing + reminder-due polling).

## Phone-to-PC API

All routes require an `X-MIDAS-Token` header matching `MIDAS_API_TOKEN`.

- `POST /chat` `{"message": "...", "history": []}` → `{"response": "..."}`
- `POST /briefing/trigger` → runs the briefing now, speaks it, returns the prose
- `GET /reminders` / `POST /reminders` `{"text": "...", "remind_at": "2026-07-14T08:00:00"}`

Bound to `127.0.0.1` by default (`config.API_HOST`). To reach it from your
phone, put it behind Tailscale and point `API_HOST` at the machine's
Tailscale IP (or `0.0.0.0` if Tailscale's ACLs are already restricting
access) — no route logic needs to change.

## Adding a new tool

1. Write a deterministic function in `tools/<name>.py`, plus a paired
   Ollama tool-schema dict next to it.
2. Import both in `main.py` and add one line to `TOOL_REGISTRY`.

No other file needs to change — `Orchestrator` builds the `tools=[...]`
list and dispatch table from `TOOL_REGISTRY` alone.
