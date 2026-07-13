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
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:3b
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
and API port. Set a real shared-secret token for the phone API before
exposing it beyond loopback:

```
set MIDAS_API_TOKEN=your-long-random-token
```

## 5. Run

```
py -3.12 main.py
```

This launches the desktop UI (main thread), starts the FastAPI server on
`127.0.0.1:8420` (background thread), and starts the APScheduler jobs
(morning briefing + reminder-due polling). Ollama must already be
running (`ollama serve`, or the Ollama desktop app) for chat, routing,
and the briefing's prose formatting to work.

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
1. Personal agent personality
2. Connect to Iphone, maybe give access to computer, connect to google account api
3. Is not aware of current news
4. Ship 