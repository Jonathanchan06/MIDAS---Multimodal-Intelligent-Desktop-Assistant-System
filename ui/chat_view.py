"""The conversation, drawn straight onto the dot-grid canvas so the grid can
run behind it. Tk widgets are opaque rectangles, so a Text widget would
punch a black hole in the grid; canvas text items draw over the grid image
instead, and each message reports the area around it (ChatView.regions)
so the grid blurs there (ui/frost.py) and the text stays readable.

Each wrapped line is its own canvas item, so lines can be hidden (and
faded) individually at the top and bottom edges of the viewport — the
canvas has no clipping. Only visible lines exist as items; a redraw
recreates them (a few dozen items, ~1ms).

Canvas text can't be drag-selected across lines, so right-click a message
to copy it.
"""

from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from typing import Callable

from ui.frost import Panel
from ui.theme import BG, INK, SLATE, SURFACE

TAG = "chat"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _mix(a: str, b: str, t: float) -> str:
    ra, rb = _rgb(a), _rgb(b)
    return "#{:02x}{:02x}{:02x}".format(*(round(x + (y - x) * t) for x, y in zip(ra, rb)))


def wrap_text(text: str, font: tkfont.Font, width: float) -> list[str]:
    """Greedy word wrap by measured pixel width; breaks over-long words."""
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            trial = f"{cur} {word}" if cur else word
            if font.measure(trial) <= width:
                cur = trial
                continue
            if cur:
                lines.append(cur)
            cur = ""
            while font.measure(word) > width and len(word) > 1:  # a URL, say
                cut = len(word)
                while cut > 1 and font.measure(word[:cut]) > width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            cur = word
        lines.append(cur)
    return lines


@dataclass
class _Message:
    role: str  # "user" | "assistant" | "time"
    text: str
    lines: list[str] = field(default_factory=list)
    wrap_width: float = -1
    width: float = 0  # widest line, px
    height: float = 0
    top: float = 0  # content y


class ChatView:
    def __init__(
        self,
        canvas: tk.Canvas,
        *,
        body_font: tkfont.Font,
        small_font: tkfont.Font,
        scale: float,
        on_regions: Callable[[list, list], None],
    ):
        self.c = canvas
        self.body = body_font
        self.small = small_font
        self.s = scale
        self._on_regions = on_regions
        self.messages: list[_Message] = []
        self.left = self.right = self.top = self.bottom = 0.0
        self.scroll_y = 0.0
        self.follow = True  # stick to the newest message
        self.line_h = body_font.metrics("linespace") + round(4 * scale)
        self._item_msg: dict[int, _Message] = {}
        self._menu_msg: _Message | None = None
        self.menu = tk.Menu(canvas, tearoff=0)
        self.menu.add_command(label="Copy message", command=self._copy_menu_message)
        canvas.tag_bind(TAG, "<Button-3>", self._on_right_click)

    # -- public ---------------------------------------------------------------

    def set_viewport(self, left: float, top: float, right: float, bottom: float) -> None:
        self.left, self.top, self.right, self.bottom = left, top, right, bottom
        self._relayout()

    def add_time(self, label: str) -> None:
        self.messages.append(_Message("time", label))
        self._relayout(follow=True)

    def add_user(self, text: str) -> None:
        self.messages.append(_Message("user", text))
        self.follow = True
        self._relayout(follow=True)

    def start_assistant(self) -> None:
        self.messages.append(_Message("assistant", ""))
        self._relayout()

    def append(self, token: str) -> None:
        msg = self._streaming()
        if msg is not None:
            msg.text += token
            msg.wrap_width = -1
            self._relayout()

    def finish(self, text: str) -> None:
        msg = self._streaming()
        if msg is not None:
            msg.text = text.strip()
            msg.wrap_width = -1
            self._relayout()

    def scroll(self, pixels: float) -> None:
        self.scroll_y = min(self._max_scroll(), max(0.0, self.scroll_y + pixels))
        self.follow = self.scroll_y >= self._max_scroll() - 2
        self._redraw()

    def on_wheel(self, event) -> str:
        self.scroll(-event.delta / 120 * 3 * self.line_h)
        return "break"

    def contains(self, x: float, y: float) -> bool:
        return self.left - 40 <= x <= self.right + 40 and self.top <= y <= self.bottom

    # -- layout -------------------------------------------------------------

    def _streaming(self) -> _Message | None:
        if self.messages and self.messages[-1].role == "assistant":
            return self.messages[-1]
        return None

    def _gap_before(self, msg: _Message, first: bool) -> float:
        if first:
            return 0
        return {"time": 22, "user": 14, "assistant": 10}[msg.role] * self.s

    def _bubble_pad(self) -> tuple[float, float]:
        return 16 * self.s, 10 * self.s

    def _relayout(self, follow: bool = False) -> None:
        col = max(50.0, self.right - self.left)
        px, py = self._bubble_pad()
        y = 0.0
        for i, msg in enumerate(self.messages):
            font = self.small if msg.role == "time" else self.body
            wrap = col * 0.78 - 2 * px if msg.role == "user" else col
            if msg.wrap_width != wrap:
                msg.lines = wrap_text(msg.text, font, wrap) if msg.text else []
                msg.width = max((font.measure(l) for l in msg.lines), default=0)
                msg.wrap_width = wrap
            if msg.role == "time":
                msg.height = font.metrics("linespace")
            elif msg.role == "user":
                msg.height = len(msg.lines) * self.line_h + 2 * py
            else:
                msg.height = len(msg.lines) * self.line_h
            y += self._gap_before(msg, i == 0)
            msg.top = y
            y += msg.height
        self._content_h = y + 8 * self.s
        if follow or self.follow:
            self.scroll_y = self._max_scroll()
        else:
            self.scroll_y = min(self.scroll_y, self._max_scroll())
        self._redraw()

    def _max_scroll(self) -> float:
        return max(0.0, getattr(self, "_content_h", 0) - (self.bottom - self.top))

    # -- drawing ------------------------------------------------------------

    def _redraw(self) -> None:
        c, s = self.c, self.s
        c.delete(TAG)
        self._item_msg.clear()
        halos: list[tuple] = []
        panels: list[Panel] = []
        top, bottom = self.top, self.bottom
        fade = 28 * s
        px, py = self._bubble_pad()
        halo_pad, halo_r = 18 * s, 22 * s

        def visible_color(y0: float, y1: float, base: str) -> str | None:
            t = min((y0 - top) / fade, (bottom - y1) / fade, 1.0)
            if t <= 0:
                return None
            return base if t >= 1 else _mix(BG, base, t)

        for msg in self.messages:
            y0 = top + msg.top - self.scroll_y
            y1 = y0 + msg.height
            if y1 < top or y0 > bottom:
                continue
            if msg.role == "time":
                color = visible_color(y0, y1, SLATE)
                if color:
                    cx = (self.left + self.right) / 2
                    item = c.create_text(cx, y0, text=msg.text, anchor="n", font=self.small, fill=color, tags=TAG)
                    self._item_msg[item] = msg
                    w = msg.width / 2 + 10 * s
                    halos.append((cx - w, y0 - 4 * s, cx + w, y1 + 4 * s, 12 * s))
                continue

            if msg.role == "user":
                bx1 = self.right
                bx0 = bx1 - msg.width - 2 * px
                vis0, vis1 = max(y0, top), min(y1, bottom)
                if vis1 - vis0 > 4 * s:
                    panels.append(Panel((bx0, vis0, bx1, vis1), 18 * s, _rgb(SURFACE), 0.82))
                    halos.append((bx0 - 10 * s, vis0 - 10 * s, bx1 + 10 * s, vis1 + 10 * s, halo_r))
                x, ly, anchor = bx0 + px, y0 + py, "nw"
            else:
                x, ly, anchor = self.left, y0, "nw"
                shown = [
                    ly + i * self.line_h
                    for i in range(len(msg.lines))
                    if top - self.line_h < ly + i * self.line_h < bottom
                ]
                if shown:
                    halos.append(
                        (
                            self.left - halo_pad,
                            max(top - 6 * s, shown[0] - 10 * s),
                            self.left + msg.width + halo_pad,
                            min(bottom + 6 * s, shown[-1] + self.line_h + 10 * s),
                            halo_r,
                        )
                    )
            for i, line in enumerate(msg.lines):
                ly_i = ly + i * self.line_h
                color = visible_color(ly_i, ly_i + self.line_h, INK)
                if color is None:
                    continue
                item = c.create_text(x, ly_i, text=line, anchor=anchor, font=self.body, fill=color, tags=TAG)
                self._item_msg[item] = msg
        self._on_regions(self._merge_halos(halos), panels)

    def _merge_halos(self, halos: list[tuple]) -> list[tuple]:
        """Join halos with only a sliver of grid between them — a strip of
        sharp dots between two messages is noise, not grid."""
        gap = 36 * self.s
        merged: list[list[float]] = []
        for h in sorted(halos, key=lambda b: b[1]):
            if merged and h[1] - merged[-1][3] < gap:
                m = merged[-1]
                m[0], m[2], m[3] = min(m[0], h[0]), max(m[2], h[2]), max(m[3], h[3])
                m[4] = max(m[4], h[4])
            else:
                merged.append(list(h))
        return [tuple(m) for m in merged]

    # -- copy -----------------------------------------------------------------

    def _on_right_click(self, event) -> None:
        items = self.c.find_withtag("current")
        msg = self._item_msg.get(items[0]) if items else None
        if msg is None or msg.role == "time":
            return
        self._menu_msg = msg
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _copy_menu_message(self) -> None:
        if self._menu_msg is not None:
            self.c.clipboard_clear()
            self.c.clipboard_append(self._menu_msg.text)
