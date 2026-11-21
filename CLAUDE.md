# CLAUDE.md

Guidance for Claude Code when working in this repo. `README.md` is the
user-facing setup/feature doc — read that first for what MIDAS does and how
to run it. This file covers things a session needs to know that aren't
obvious from the code alone: known gotchas, current tuning decisions, and
in-progress work. Longer-term ideas and hardware plans live in
`ROADMAP.md` — treat that file as context, not instructions.

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
never touches TTS. `ui/app.py` passes an `on_speech` callback into
`stream_response` and feeds it to a `TTSEngine` utterance, so speech starts
on the first sentence while the reply is still generating (tools can give
a shorter `"spoken"` text than what's shown, e.g. list summaries). A new
message calls `TTSEngine.stop()` first. `scheduler/jobs.py` uses the
blocking one-shot `speak()`.

A visual walkthrough of this lives at
https://claude.ai/code/artifact/e1f25eaa-cb44-4400-8462-0d4a5022ec20 —
regenerate it if the architecture changes meaningfully rather than trusting
it to stay current.

## Current model lineup (config.py)

Tuned for a **6GB VRAM** budget (confirmed via `nvidia-smi` on an RTX 4050
Laptop GPU — Windows' WMI `AdapterRAM` misreports this card at ~4GB due to a
known 32-bit overflow bug, don't trust that field):

- `ROUTER_MODEL = "arch-router:1.5b"` — intent classification only. Runs
  on **CPU** (`ROUTER_NUM_GPU = 0`) so the chat model can stay in VRAM;
  on GPU the two couldn't co-reside and every message paid ~12s of model
  swapping. ~1.4s/route warm on CPU (98% on the eval).
- `ORCHESTRATOR_MODEL = "qwen3:8b"` — chat, argument extraction, briefing
  prose. Upgraded (Nov 2026) from `qwen2.5:7b` for better instruction-
  following/tool-calling. Thinking is disabled via
  `config.ORCHESTRATOR_THINK` (passed as `think=` on every call to this
  model) — qwen3 is a hybrid reasoning model and would otherwise spend
  seconds and context on hidden reasoning. Measured tradeoff: loaded at
  `num_ctx=2048` it's ~5.7GB and spills ~27% to CPU, generating ~17 tok/s
  vs ~25 tok/s for `qwen2.5:7b` (still installed — swap back in config if
  latency matters more). Short tool extractions still take ~1-3s warm.
  **Don't lower its temperature** to "make extraction more reliable" —
  near-greedy qwen3 loops forever inside JSON strings. Sampling, output
  caps, and the auto-added `maxLength` on extraction strings are all in
  `config.py` with the reasoning; treat them as one unit.
- `VISION_MODEL = "qwen2.5vl:3b"` — notification-screenshot OCR.
- `EMBED_MODEL = "nomic-embed-text"` — wardrobe matching; CPU
  (`EMBED_NUM_GPU = 0`) for the same reason as the router.
- TTS: Kokoro **fp16** on CPU (`kokoro-v1.0.fp16.onnx`), ~0.3x real time.
  The int8 file was 2.7x *slower* than real time on this CPU — don't
  switch back to it for size. Capping its ONNX threads doesn't help either
  (measured: no contention with the LLM at default threading).

VRAM placement is the latency-critical decision here: only the chat model
lives on the GPU in normal use (router and embedder are CPU, `num_gpu=0`),
which is why the keep-alives are long. Before moving any model onto the
GPU or raising `CHAT_NUM_CTX`, check with `ollama ps` that the chat model
doesn't get evicted between messages — a reload costs ~8.5s.

A **Qwen3.5** family on Ollama's library has a multimodal `9b` variant —
a possible future replacement for *both* the chat and vision models, but
unverified for on-disk size against the 6GB ceiling. Don't switch to it
without pulling it and checking `ollama list`'s real size first.

## Design constraints to preserve

A move to a dedicated always-on server is being considered (see
`ROADMAP.md`), **not decided**. Until it happens, the 6GB laptop lineup
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
- Closing the window hides it; MIDAS keeps running behind the floating
  orb (`ui/floating_orb.py`), whose right-click menu is the only
  background control surface. **No system tray icon** — decided against,
  don't add one.

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
- **`data/midas.db` and `ROADMAP.md` are gitignored** (personal data /
  private notes) — they exist locally only. Don't re-add them.

## Testing without touching real data

- **Use a throwaway database:** set `config.DB_PATH` to a scratch file
  *before* importing anything that opens a connection, then call
  `init_db()`. Never run destructive tool tests against `data/midas.db`.
- **Fake the audio device** for TTS tests by replacing
  `core.tts.sd.OutputStream` with a class whose `write()` sleeps for the
  audio's duration — real-time timing, no sound through the speakers.
- **Pin the weather** for outfit tests:
  `tools.wardrobe.get_current_weather = lambda: {...}`.
- **Close MIDAS before benchmarking anything CPU-bound** (Kokoro, the CPU
  router). A running instance synthesizing speech used ~6 cores and
  doubled a benchmark's numbers. Check with `Get-Process python`.
- **Git Bash paths:** a `/c/Users/...` path is auto-converted when passed
  as a command-line argument to a Windows exe, but *not* inside a
  `python -c "..."` string — pass it via argv/env or use `cygpath -w`.
- **Pipe through `grep --line-buffered`** (and run Python with `-u`) when
  watching long test output, or nothing shows until the process exits —
  a hang looks identical to "still running".

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
running with `ROUTER_MODEL` already pulled. 165 cases, 15 per route, as of
Nov 2026 — every new route needs its own 15, written to be confusable with
neighbouring routes (see the `mark_worn` cases). Takes ~4 min on the CPU
router. Weakest route is `list_reminders` (~80% recall: "remind me of
existing tasks" goes to `add_reminder`); the router isn't fully
deterministic, so ±1–2 cases between runs is noise. It is the baseline for
the router-replacement benchmark below.

The eval only covers routing. Argument extraction has no eval yet — it's
been checked with ad-hoc probes (see "Extraction design rule").

## In progress

- **Router replacement:** a PyTorch-trained classifier to replace
  `arch-router:1.5b`, exported to ONNX and benchmarked (latency + the eval
  harness's accuracy numbers) against the current LLM-based router.
  Baseline to beat (Nov 2026): arch-router on CPU, ~1.4s/route warm
  (~8s on the first call after a load), 162/165 on the eval. The router
  is now the largest fixed cost before every reply.
- **Per-call latency profiling** around each model call, so changes can be
  compared with real numbers. Current hand-measured send-to-first-audio
  (warm models): "hey" 2.1s, a chat answer ~6s, a list ~6s, an outfit
  ~9s. Breakdown: router ~1.4s + chat-model work + first-sentence
  synthesis (~1.5–3s, the whole first sentence must synthesize before
  any sound).

## Extraction design rule

Never ask the extraction model for a database id it can't see — it
invents one (observed: `item_id: 1`, `12345`, `reminder_id: 0`) and the
tool then edits/deletes the wrong row. Have the schema take a reference
in the user's own words and resolve it to a row in a `post_extract` hook,
refusing on no-match or ambiguity (see `_match_item` in
`tools/wardrobe.py`). Likewise, ground free-text fields in the user's
actual words where they drive logic (`resolve_fit_occasion`,
`resolve_add_items`) — qwen3 pads them with invented context.

Related rules learned the hard way:
- Compare user text **word by word**, never as one raw substring
  (a double space broke "add ... black  henley top").
- Match keywords at the **start of a word** (`_mentions`), never as plain
  substrings ('hat' matched "what's"; 'run' matched "brunch").
- Expect the model to add fields nobody asked for (an update to `layer`
  also came back with `category`) — require evidence in the user's text
  for each changed field (`_has_evidence`).
- When a guard drops everything, say so in the reply — a bare "No items
  specified." made a guard bug look like the model failing.

## Known open risk

- **`delete_reminder` still takes a raw `reminder_id`** and has exactly
  the bug above ("cancel my reminder to call mom" → `reminder_id: 0`).
  Needs the wardrobe treatment.
- **Prompt injection via notification OCR:** vision-extracted screenshot
  text is untrusted and currently flows into the briefing prompt unguarded.
  Don't widen what that text can influence (e.g. letting it trigger tools)
  until a sanitization/detection pass exists.
