# CLAUDE.md

Guidance for Claude Code when working in this repo. `README.md` is the
user-facing setup/feature doc — read that first for what MIDAS does and how
to run it. This file covers things a session needs to know that aren't
obvious from the code alone: known gotchas, current tuning decisions, and
in-progress work. Longer-term ideas and hardware plans live in
`docs/ROADMAP.md` — treat that file as context, not instructions.

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

A visual walkthrough of this lives at
https://claude.ai/code/artifact/e1f25eaa-cb44-4400-8462-0d4a5022ec20 —
regenerate it if the architecture changes meaningfully rather than trusting
it to stay current.

## Current model lineup (config.py)

Tuned for a **6GB VRAM** budget (confirmed via `nvidia-smi` on an RTX 4050
Laptop GPU — Windows' WMI `AdapterRAM` misreports this card at ~4GB due to a
known 32-bit overflow bug, don't trust that field):

- `ROUTER_MODEL = "arch-router:1.5b"` — intent classification only.
- `ORCHESTRATOR_MODEL = "qwen3:8b"` — chat, argument extraction, briefing
  prose. Upgraded (Nov 2026) from `qwen2.5:7b` for better instruction-
  following/tool-calling at the same ~5GB footprint; same size class, newer
  generation, not a bigger model (VRAM ceiling didn't change).
- `VISION_MODEL = "qwen2.5vl:3b"` — notification-screenshot OCR.

Short `keep_alive`s are deliberate so idle models evict fast and don't
starve whichever model runs next — don't "fix" this by lengthening them
without re-checking the VRAM math.

A **Qwen3.5** family on Ollama's library has a multimodal `9b` variant —
a possible future replacement for *both* the chat and vision models, but
unverified for on-disk size against the 6GB ceiling. Don't switch to it
without pulling it and checking `ollama list`'s real size first.

## Design constraints to preserve

A move to a dedicated always-on server is being considered (see
`docs/ROADMAP.md`), **not decided**. Until it happens, the 6GB laptop lineup
above is the real target — don't tune models, `keep_alive`s, or context
sizes for hardware that doesn't exist yet. But don't make the move harder:

- Don't hardcode `localhost:11434` or assume Ollama runs on the same machine
  as the backend; read the host from `config.py`.
- Keep `ui/app.py` thin — anything that isn't UI belongs behind the
  Orchestrator / API, so the desktop app can later become a client of a
  backend running elsewhere.
- Keep model choices swappable via config rather than baked into code
  (future: per-hardware profiles).
- The Orchestrator stays the only caller of Ollama.

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
- **Merge-conflict markers have been committed into `.py` files before**
  (from the `85ab153`/`b4bdb17` merge; fixed Nov 2026). They're a
  `SyntaxError` on import, not a lint warning. Resolve conflicts by reading
  both sides — one is often a stale pre-feature version, and methods can get
  tangled together (as `format_briefing`/`extract_notification_text` did),
  not just duplicated.

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
running with `ROUTER_MODEL` already pulled. The test set is still small
(~30 cases as of Nov 2026) — keep growing it, especially tricky/paraphrased
cases and full `delete_reminder` coverage, not just the obvious ones. It is
the baseline for the router-replacement benchmark below.

## In progress

- **Router replacement:** a PyTorch-trained classifier to replace
  `arch-router:1.5b`, exported to ONNX and benchmarked (latency + the eval
  harness's accuracy numbers) against the current LLM-based router.
- **Per-call latency profiling** around each model call, so changes can be
  compared with real numbers.
- **Run in the background instead of quitting on window-close.** Main thread
  is owned by the CustomTkinter window (`ui/app.py`); scheduler + FastAPI
  already run on background/daemon threads.
  - **Done: floating orb** (`ui/floating_orb.py`, drawing shared with the
    window's orb via `ui/orb.py`). Built first, standalone, instead of
    after the tray icon — the orb's right-click menu carries what was
    planned for the tray menu (Open/Hide / Run briefing now / Mute voice /
    Quit), so closing the window now hides it rather than exiting. Live
    idle/thinking/speaking states come from `TTSEngine.add_listener` (so
    scheduler-fired speech lights it up too, not just chat). Edge snapping
    is per-monitor work area via `MonitorFromPoint`; position persists in
    `data/orb_position.json`. Single-instance mutex is in `main.py`.
    Deliberately decoupled from orchestrator/TTS (callbacks +
    `set_state()` only) so it can become ROADMAP.md's thin client.
  - **Still to do: system tray icon** — `pystray` + Pillow, icon drawn at
    runtime echoing the orb design, same menu as the orb. Mainly a
    fallback if the orb gets dragged/hidden behind a fullscreen app, plus
    Windows toast notifications (`icon.notify()`) for reminders.
  - **Done: headless launch.** `venv\Scripts\pythonw.exe main.py
    --background` runs with no console and starts as just the orb (README
    "Run in the background" has the start-at-login shortcut). Logging goes
    to `data/midas.log`; see `_configure_logging` in `main.py` for why
    uvicorn needs `log_config=None` under pythonw.
  - **Gotchas:** testing pythonw from Git Bash hides the stdout/stderr-is-
    `None` problem (it inherits the shell's pipes) — launch via
    PowerShell `Start-Process` to reproduce a real shortcut launch. The
    venv's `pythonw.exe` is a launcher that spawns the real interpreter as
    a *child* process, so a window/PID check must look at the child.
    `pystray` callbacks run off the main thread — route them through the
    existing `_queue`/`_poll_queue` pattern, don't touch Tkinter widgets
    directly from them. Quit (`MidasApp.quit_app`) should also stop the
    tray icon once it exists.
  - Once shipped, move these gotchas into code comments next to the code
    they apply to and trim this section.
  - Packaging as a standalone `.exe` (the README's "Ship" item) can wait —
    bundling `onnxruntime`/`kokoro-onnx`'s data files is fiddly, and
    `pythonw` + a startup shortcut gets most of the same experience sooner.

## Known open risk

- **Prompt injection via notification OCR:** vision-extracted screenshot
  text is untrusted and currently flows into the briefing prompt unguarded.
  Don't widen what that text can influence (e.g. letting it trigger tools)
  until a sanitization/detection pass exists.
