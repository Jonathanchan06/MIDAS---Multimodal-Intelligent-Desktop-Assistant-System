"""CustomTkinter port of the imported design (Local LLM Agent.dc.html):
a centered "orb" status indicator over a single scrollable transcript,
light background, cyan accent. The web mock's continuous mouse-tracked
gradient/parallax is dropped in favor of a cheap Enter/Leave hover state —
a full-canvas redraw on every mouse-move event would fight the "keep it
fast" hardware constraint for no real benefit on a desktop app.

Generation runs on a daemon thread; tokens flow back through a
queue.Queue drained by after()-polling, so the LLM call never touches
Tkinter widgets off the main thread and the window never freezes.

Closing the window hides it instead of quitting — the floating orb
(ui/floating_orb.py) stays on screen as the status light and the way back
in, and the scheduler + API keep running. Quit lives in the orb's
right-click menu.
"""

import logging
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime

import customtkinter as ctk

import config
from tools.briefing import run_morning_briefing
from ui.floating_orb import FloatingOrb
from ui.orb import IDLE, SPEAKING, THINKING, OrbDrawing
from ui.theme import (
    ACCENT,
    BG,
    BORDER,
    TEXT_FAINT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_USER,
)

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 12  # keep prompts small for the 2048-token chat context budget


def _pick_font(preferred: str, fallback: str) -> str:
    try:
        available = set(tkfont.families())
    except Exception:
        return fallback
    return preferred if preferred in available else fallback


class MidasApp(ctk.CTk):
    def __init__(self, orchestrator, tts_engine):
        super().__init__()
        self.orchestrator = orchestrator
        self.tts_engine = tts_engine
        self.history: list[dict] = []
        self.is_streaming = False
        self._queue: "queue.Queue" = queue.Queue()
        self._streaming_label = None
        self._streaming_text = ""
        self._speaking = False

        ctk.set_appearance_mode("light")

        self.title("MIDAS")
        self.geometry("760x900")
        self.configure(fg_color=BG)

        body = _pick_font("Space Grotesk", "Segoe UI")
        mono = _pick_font("IBM Plex Mono", "Consolas")
        self.font_body = ctk.CTkFont(family=body, size=15)
        self.font_title = ctk.CTkFont(family=body, size=13, weight="bold")
        self.font_msg_label = ctk.CTkFont(family=body, size=11)
        self.font_mono = ctk.CTkFont(family=mono, size=10)
        self.font_mono_xs = ctk.CTkFont(family=mono, size=9)
        self.font_send = ctk.CTkFont(family=body, size=16)

        self._build_top_bar()
        self._build_orb()
        self._build_transcript()
        self._build_input_row()

        self.mute_var = tk.BooleanVar(value=False)
        self.mute_var.trace_add(
            "write", lambda *_: setattr(self.tts_engine, "muted", self.mute_var.get())
        )
        self.floating_orb = FloatingOrb(
            self,
            on_toggle_window=self.toggle_window,
            on_run_briefing=self.run_briefing_now,
            on_quit=self.quit_app,
            is_window_visible=self.is_window_visible,
            mute_var=self.mute_var,
        )
        self.protocol("WM_DELETE_WINDOW", self.hide_window)
        # Fires on the TTS thread — hand off through the queue, never touch
        # widgets from there.
        self.tts_engine.add_listener(lambda speaking: self._queue.put(("speaking", speaking)))

        self._animate_orbs()
        self._poll_queue()

    # -- layout -----------------------------------------------------------

    def _build_top_bar(self):
        bar = ctk.CTkFrame(self, fg_color=BG, height=40, corner_radius=0)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        ctk.CTkLabel(
            bar, text="MIDAS · LOCAL RUNTIME", font=self.font_mono, text_color=TEXT_MUTED
        ).place(x=28, rely=0.5, anchor="w")

        right = ctk.CTkFrame(bar, fg_color=BG)
        right.place(relx=1.0, x=-28, rely=0.5, anchor="e")
        dot = tk.Canvas(right, width=8, height=8, bg=BG, highlightthickness=0)
        dot.create_oval(1, 1, 7, 7, fill=ACCENT, outline=ACCENT)
        dot.pack(side="left", padx=(0, 6))
        self.model_label = ctk.CTkLabel(
            right, text=config.ORCHESTRATOR_MODEL, font=self.font_mono, text_color=TEXT_MUTED
        )
        self.model_label.pack(side="left")

    def _build_orb(self):
        wrap = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        wrap.pack(pady=(44, 0))

        self.orb_canvas = tk.Canvas(
            wrap, width=168, height=168, bg=BG, highlightthickness=0, cursor="hand2"
        )
        self.orb_canvas.pack()
        self.orb = OrbDrawing(self.orb_canvas, 168)
        self.orb_canvas.bind("<Enter>", lambda _e: self.orb.set_hover(True))
        self.orb_canvas.bind("<Leave>", lambda _e: self.orb.set_hover(False))

        ctk.CTkLabel(wrap, text="MIDAS", font=self.font_title, text_color=TEXT_PRIMARY).pack(
            pady=(18, 4)
        )
        self.status_label = ctk.CTkLabel(
            wrap, text="READY", font=self.font_mono, text_color=TEXT_MUTED
        )
        self.status_label.pack()

    def _build_transcript(self):
        self.transcript = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        self.transcript.pack(fill="both", expand=True, pady=(32, 12))

        self.column = ctk.CTkFrame(self.transcript, fg_color=BG, corner_radius=0)
        self.column.pack(anchor="center")

    def _build_input_row(self):
        outer = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        outer.pack(fill="x", side="bottom", pady=(0, 28))

        column = ctk.CTkFrame(outer, fg_color=BG, corner_radius=0)
        column.pack(anchor="center")

        entry_row = ctk.CTkFrame(column, fg_color=BG, corner_radius=0)
        entry_row.pack(fill="x")

        # CTkEntry (single line) stands in for the web mock's auto-growing
        # textarea — CTk has no multi-line widget with that grow behavior.
        self.entry = ctk.CTkEntry(
            entry_row,
            placeholder_text="speak or type to MIDAS...",
            font=self.font_body,
            fg_color=BG,
            border_width=0,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_FAINT,
            width=560,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.entry.bind("<Return>", self._on_send)
        self.entry.bind("<FocusIn>", lambda e: self.border_line.configure(fg_color=ACCENT))
        self.entry.bind("<FocusOut>", lambda e: self.border_line.configure(fg_color=BORDER))

        self.send_button = ctk.CTkButton(
            entry_row,
            text="→",
            width=28,
            font=self.font_send,
            fg_color=BG,
            hover_color=BG,
            text_color=ACCENT,
            corner_radius=0,
            command=self._on_send,
        )
        self.send_button.pack(side="right")

        self.border_line = ctk.CTkFrame(column, fg_color=BORDER, height=1, corner_radius=0)
        self.border_line.pack(fill="x", pady=(10, 0))

        ctk.CTkLabel(
            column,
            text="RUNS OFFLINE · NOTHING LEAVES THIS DEVICE",
            font=self.font_mono_xs,
            text_color=TEXT_FAINT,
        ).pack(pady=(10, 0))

    # -- orbs / activity state -------------------------------------------------

    def _animate_orbs(self):
        # The window's own orb is skipped while hidden — no point redrawing
        # an invisible canvas every 60ms. The floating orb always animates.
        if self.is_window_visible():
            self.orb.tick()
        self.floating_orb.tick()
        self.after(60, self._animate_orbs)

    def _refresh_activity(self):
        if self._speaking:
            state, label = SPEAKING, "SPEAKING…"
        elif self.is_streaming:
            state, label = THINKING, "THINKING…"
        else:
            state, label = IDLE, "READY"
        self.orb.set_state(state)
        self.floating_orb.set_state(state)
        self.status_label.configure(text=label)

    # -- window visibility --------------------------------------------------------

    def is_window_visible(self) -> bool:
        return self.state() not in ("withdrawn", "iconic")

    def hide_window(self):
        self.withdraw()
        self.floating_orb.raise_to_top()

    def show_window(self):
        self.deiconify()
        self.lift()
        # Windows won't always let a background process steal foreground;
        # a brief topmost toggle reliably brings the window forward.
        self.attributes("-topmost", True)
        self.after_idle(self.attributes, "-topmost", False)
        self.focus_force()
        self.entry.focus_set()
        self.floating_orb.raise_to_top()

    def toggle_window(self):
        if self.is_window_visible():
            self.hide_window()
        else:
            self.show_window()

    def quit_app(self):
        # mainloop() returns after destroy(); main.py's finally block then
        # shuts the scheduler down. API/worker threads are daemons.
        self.tts_engine.stop()
        self.floating_orb.destroy()
        self.destroy()

    # -- briefing on demand ---------------------------------------------------------

    def run_briefing_now(self):
        """Orb-menu "Run briefing now". Goes into the transcript (and
        history) as if the user had asked for it, so follow-ups like
        "tell me more about that headline" have context. Doesn't pop the
        window open — it's spoken either way."""
        if self.is_streaming:
            return
        request = "Give me my morning briefing."
        timestamp = datetime.now().strftime("%H:%M")
        self._add_message("user", request, timestamp)
        self._start_assistant_turn(timestamp)
        threading.Thread(target=self._briefing_worker, args=(request,), daemon=True).start()

    def _briefing_worker(self, request: str):
        try:
            prose = self.orchestrator.format_briefing(run_morning_briefing())
        except Exception as exc:
            logger.exception("on-demand briefing failed")
            self._queue.put(("done", (request, f"[error: {exc}]")))
            return
        self._queue.put(("done", (request, prose)))
        self.tts_engine.speak(prose)

    # -- transcript -----------------------------------------------------------

    def _add_message(self, role: str, content: str, timestamp: str):
        is_user = role == "user"
        row = ctk.CTkFrame(self.column, fg_color=BG, corner_radius=0)
        row.pack(fill="x", pady=(0, 26))

        header = ctk.CTkFrame(row, fg_color=BG, corner_radius=0)
        header.pack(anchor="e" if is_user else "w")
        ctk.CTkLabel(
            header,
            text="YOU" if is_user else "MIDAS",
            font=self.font_msg_label,
            text_color=TEXT_MUTED if is_user else ACCENT,
        ).pack(side="left")
        ctk.CTkLabel(
            header, text=f"  {timestamp}", font=self.font_mono_xs, text_color=TEXT_FAINT
        ).pack(side="left")

        content_label = ctk.CTkLabel(
            row,
            text=content,
            font=self.font_body,
            text_color=TEXT_USER if is_user else TEXT_PRIMARY,
            wraplength=600,
            justify="right" if is_user else "left",
            anchor="e" if is_user else "w",
        )
        content_label.pack(fill="x", pady=(5, 0))
        self._scroll_to_bottom()
        return content_label

    def _scroll_to_bottom(self):
        self.update_idletasks()
        try:
            self.transcript._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    # -- send / streaming ---------------------------------------------------

    def _on_send(self, event=None):
        text = self.entry.get().strip()
        if not text or self.is_streaming:
            return "break"
        self.entry.delete(0, "end")

        timestamp = datetime.now().strftime("%H:%M")
        self._add_message("user", text, timestamp)
        self._start_assistant_turn(timestamp)

        history_snapshot = list(self.history)
        threading.Thread(
            target=self._worker, args=(text, history_snapshot), daemon=True
        ).start()
        return "break"

    def _start_assistant_turn(self, timestamp: str):
        self._streaming_label = self._add_message("assistant", "", timestamp)
        self._streaming_text = ""
        self.is_streaming = True
        self.send_button.configure(text="···", text_color=TEXT_FAINT, state="disabled")
        self._refresh_activity()

    def _worker(self, text: str, history_snapshot: list[dict]):
        def on_token(token: str):
            self._queue.put(("token", token))

        def on_route(model: str):
            self._queue.put(("route", model))

        try:
            full_text = self.orchestrator.stream_response(
                text, history_snapshot, on_token, on_route
            )
        except Exception as exc:
            full_text = f"[error: {exc}]"

        self._queue.put(("done", (text, full_text)))

        # This thread is already off the Tkinter main thread, so blocking
        # here on playback is safe.
        if full_text:
            self.tts_engine.speak(full_text)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "token":
                    self._streaming_text += payload
                    if self._streaming_label is not None:
                        self._streaming_label.configure(text=self._streaming_text)
                    self._scroll_to_bottom()
                elif kind == "route":
                    self.model_label.configure(text=payload)
                elif kind == "speaking":
                    self._speaking = payload
                    self._refresh_activity()
                elif kind == "done":
                    user_text, full_text = payload
                    if self._streaming_label is not None:
                        self._streaming_label.configure(text=full_text)
                    self.history.append({"role": "user", "content": user_text})
                    self.history.append({"role": "assistant", "content": full_text})
                    self.history = self.history[-HISTORY_LIMIT:]
                    self.is_streaming = False
                    self._streaming_label = None
                    self.send_button.configure(text="→", text_color=ACCENT, state="normal")
                    self._refresh_activity()
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)
