"""The orb drawing, shared by the main window's centered orb and the
floating desktop orb so both render the same design at any size and show
the same live status: idle / thinking / speaking.

Pure canvas drawing — no app, orchestrator, or TTS knowledge. Whoever owns
the canvas calls set_state()/set_hover() on changes and tick() from its
own after()-loop; this class never schedules anything itself.
"""

import math

from ui.theme import ACCENT, BORDER_STRONG, CORE_FILL, RING_DASHED, RING_INNER

# Every coordinate below was designed against the main window's 168px orb;
# other sizes scale proportionally from it.
_REFERENCE_SIZE = 168

IDLE, THINKING, SPEAKING = "idle", "thinking", "speaking"

# Per-state animation speed: (dash offset step per tick, pulse phase step per tick).
_SPEEDS = {IDLE: (1, 0.0), THINKING: (3, 0.15), SPEAKING: (1, 0.35)}


def _lerp_color(a: str, b: str, t: float) -> str:
    ar, ag, ab = (int(a[i : i + 2], 16) for i in (1, 3, 5))
    br, bg, bb = (int(b[i : i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(
        round(ar + (br - ar) * t), round(ag + (bg - ag) * t), round(ab + (bb - ab) * t)
    )


class OrbDrawing:
    def __init__(self, canvas, size: int, disc_fill: str = ""):
        """disc_fill paints the outer circle solid — the floating orb needs
        that to stay visible over arbitrary wallpaper; the main window's
        orb sits on a white background already and leaves it empty."""
        self.canvas = canvas
        s = size / _REFERENCE_SIZE
        self._scale = s
        self._center = size / 2
        self._core_radius = 28 * s
        self._hover_grow = max(2.0, 3 * s)
        self._state = IDLE
        self._hover = False
        self._dash_offset = 0
        self._phase = 0.0

        def ring(inset: float, **kwargs):
            return canvas.create_oval(inset, inset, size - inset, size - inset, **kwargs)

        ring(1 * s, outline=BORDER_STRONG, width=1, fill=disc_fill)
        self.middle_ring_id = ring(10 * s, outline=RING_DASHED, width=1, dash=(1, 4))
        self.inner_ring_id = ring(21 * s, outline=RING_INNER, width=1)
        self.core_id = canvas.create_oval(0, 0, 0, 0, fill=CORE_FILL, outline="")
        self._redraw()

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._phase = 0.0
        self._redraw()

    def set_hover(self, hover: bool) -> None:
        self._hover = hover
        self._redraw()

    def tick(self) -> None:
        dash_step, phase_step = _SPEEDS[self._state]
        self._dash_offset = (self._dash_offset + dash_step) % 1000
        self._phase += phase_step
        try:
            self.canvas.itemconfigure(self.middle_ring_id, dashoffset=self._dash_offset)
        except Exception:
            pass
        if phase_step:
            self._redraw()

    def _redraw(self) -> None:
        pulse = (math.sin(self._phase) + 1) / 2  # 0..1
        radius = self._core_radius
        if self._state == THINKING:
            fill = _lerp_color(CORE_FILL, RING_INNER, pulse)
            ring_color = ACCENT
        elif self._state == SPEAKING:
            fill = ACCENT
            ring_color = RING_INNER
            radius *= 1 + 0.12 * pulse
        else:
            fill = CORE_FILL
            ring_color = RING_DASHED

        if self._hover:
            radius += self._hover_grow
            outline, width = ACCENT, 2
        else:
            outline, width = "", 1

        c = self._center
        self.canvas.coords(self.core_id, c - radius, c - radius, c + radius, c + radius)
        self.canvas.itemconfigure(self.core_id, fill=fill, outline=outline, width=width)
        self.canvas.itemconfigure(self.middle_ring_id, outline=ring_color)
