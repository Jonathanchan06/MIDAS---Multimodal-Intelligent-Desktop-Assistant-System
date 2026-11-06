# CLAUDE.md

Guidance for Claude Code when working in this repo. `README.md` is the
user-facing setup/feature doc — read that first for what MIDAS does and how
to run it. This file covers things a session needs to know that aren't
obvious from the code alone: known gotchas, current tuning decisions, and
in-progress work.

## Architecture, in one paragraph

Three entry points — `ui/app.py` (desktop, direct in-process calls),
`server/api.py` (FastAPI, phone over Tailscale), and `scheduler/jobs.py`
(APScheduler cron/interval jobs) — all call into `core/orchestrator.py`'s
`Orchestrator`, the only thing that ever talks to Ollama. It classifies
intent with a small router model, dispatches to a deterministic tool in
`tools/*.py` (backed by SQLite via `core/db.py`), or streams a plain chat
reply. A separate vision model handles overnight notification-screenshot
OCR. Not every caller goes through the router: FastAPI's `/reminders` and
`/notification/latest` already know exactly which operation they want and
call tools directly, skipping classification entirely. The Orchestrator
never touches TTS — `ui/app.py` and `scheduler/jobs.py` each call
`core/tts.py`'s `TTSEngine` themselves after they already have text back.

A full visual walkthrough of this (built this session) lives at
https://claude.ai/artifact/UuFXKNgtpp6RG1P9PPgYVu — regenerate it if the
architecture changes meaningfully rather than trusting it to stay current.

## Current model lineup (config.py)

Tuned for a **6GB VRAM** budget (confirmed via `nvidia-smi` on an RTX 4050
Laptop GPU — Windows' WMI `AdapterRAM` misreports this card at ~4GB due to a
known 32-bit overflow bug, don't trust that field):

- `ROUTER_MODEL = "arch-router:1.5b"` — intent classification only.
- `ORCHESTRATOR_MODEL = "qwen3:8b"` — chat, argument extraction, briefing
  prose. Upgraded this session from `qwen2.5:7b` for better instruction-
  following/tool-calling at the same ~5GB footprint; same size class, newer
  generation, not a bigger model (VRAM ceiling didn't change).
- `VISION_MODEL = "qwen2.5vl:3b"` — notification-screenshot OCR.

Short `keep_alive`s are deliberate so idle models evict fast and don't
starve whichever model runs next — don't "fix" this by lengthening them
without re-checking the VRAM math.

A **Qwen3.5** family exists on Ollama's library (newer than this file's
knowledge) with a multimodal `9b` variant — interesting as a possible future
replacement for *both* the chat and vision models, but unverified for
on-disk size against the 6GB ceiling. Don't switch to it without pulling it
and checking `ollama list`'s real size first.

## Known gotchas

- **Windows' `py` launcher may not have 3.12 registered** even though the
  README asks for 3.12+ — check `py -0` first. The project's own
  `venv\Scripts\python.exe` is a valid fallback (confirmed at 3.13, which
  satisfies "3.12+") if the launcher can't find a matching version.
- **Running `eval/` scripts directly** (not via `python -m`) won't find
  `core`/`main` on `sys.path` unless the script inserts the project root
  itself — see the top of `eval/router_eval.py` for the fix, replicate it in
  any new script under `eval/`.
- **Ollama pulls can fail with "not enough space on disk"** mid-download,
  leaving a `*-partial` blob in `~/.ollama/models/blobs/` that should be
  deleted before retrying. If space is tight, redirect storage via
  `setx OLLAMA_MODELS "D:\...\"` and fully restart Ollama (quit from the
  tray, relaunch) to pick it up.
- **`MIDAS_API_TOKEN` is set via `setx`**, which only applies to new
  processes — a terminal open before you ran `setx` won't see it. Never put
  the actual token value in this file or in a commit; if you need to find
  the real one currently in use, check the `X-MIDAS-Token` header inside an
  existing iOS Shortcut rather than the env var, since that's the value
  that's actually been authenticating.
- **This repo has had real merge-conflict markers (`<<<<<<<`/`=======`/
  `>>>>>>>`) committed directly into `.py` files before** (`config.py`,
  `core/orchestrator.py`, `server/api.py`, `README.md`, from the
  `85ab153`/`b4bdb17` merge) — this is a `SyntaxError` the moment Python
  tries to import the file, not a lint warning. Fixed as of this session;
  if a future merge reintroduces this, resolve it by actually reading both
  sides (one side is often a stale pre-feature version, not a real
  alternative) rather than blindly picking one — the `format_briefing`/
  `extract_notification_text` conflict specifically had gotten its two
  methods' bodies tangled together, not just duplicated text.

## Eval harness (`eval/`)

`eval/router_eval.py` runs `Orchestrator.route()` against labeled cases in
`eval/test_cases.json` and reports accuracy, per-route precision/recall, and
a confusion matrix. Run it after touching any route description, swapping
`ROUTER_MODEL`, or changing the router prompt — it's the only objective
signal for whether that change helped or hurt, instead of eyeballing a few
chat messages.

```
python eval/router_eval.py
```

(or the venv's `python.exe` directly, per the gotcha above). Requires Ollama
running with `ROUTER_MODEL` already pulled. The test set is intentionally
thin right now (a handful of cases per route, as of this session) — it's
meant to keep growing, especially tricky/paraphrased cases and full
`delete_reminder` coverage, not just the obvious ones.

## In-progress / discussed, not yet built

- A PyTorch-trained classifier to replace `arch-router:1.5b` for routing,
  exported to ONNX and benchmarked (latency + the eval harness's accuracy
  numbers) against the current LLM-based router. The eval harness above is
  the prerequisite ground truth for that comparison.
- Local STT (`faster-whisper`/`whisper.cpp`) for a full voice-in loop — TTS
  already exists one-way, nothing listens yet.
- A sanitization/detection pass on vision-extracted notification text before
  it reaches the briefing prompt — real prompt-injection attack surface
  already exists there (untrusted screenshot content → LLM), just unguarded.
- LoRA fine-tune for the "personal agent personality" item in the README's
  future-additions list.
- **Run in the background instead of quitting on window-close.** Main thread
  is owned by the CustomTkinter window (`ui/app.py`); scheduler + FastAPI
  already run on background/daemon threads. Plan, in order:
  - **Option A (do first): system tray icon** — `pystray` + Pillow, icon
    drawn at runtime (no asset file) echoing the orb design. Close button
    calls `self.withdraw()` instead of exiting; tray menu: Open MIDAS / Run
    briefing now / Mute voice / Quit. Reminders could also fire as Windows
    toast notifications (`icon.notify()`) alongside speech.
  - **Option B (later, stacks on A): floating orb** docked to a screen
    edge — borderless always-on-top `CTkToplevel`
    (`overrideredirect`/`-topmost`/`-transparentcolor`), reusing the
    existing `_build_orb` drawing code as a live idle/thinking/speaking
    status light. More fiddly (edge snapping, multi-monitor).
  - **Gotchas to handle:** `pystray` callbacks run off the main thread —
    route them through the existing `_queue`/`_poll_queue` pattern, don't
    touch Tkinter widgets directly from them. Launching via `pythonw.exe`
    (no console) makes `sys.stdout`/`stderr` `None`, which crashes
    uvicorn's default logging — pass `log_config=None` and log to a file
    (e.g. `data/midas.log`) instead. Guard against a second instance (e.g.
    a named Windows mutex via `ctypes`) or you'll get a silent port-8420
    failure and the morning briefing firing twice. Pause `_animate_orb`
    while the window is withdrawn so it's not redrawing every 60ms for
    nothing. Quit should stop the tray icon, call `scheduler.shutdown()`,
    then `app.destroy()`. Start-at-login via a `pythonw.exe main.py`
    shortcut in `shell:startup` or an `HKCU\...\Run` key.
  - Packaging as a standalone `.exe` (the README's "Ship" item) can wait —
    bundling `onnxruntime`/`kokoro-onnx`'s data files is fiddly, and
    `pythonw` + a startup shortcut gets most of the same experience sooner.
