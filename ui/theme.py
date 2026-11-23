"""Shared palette and type for the main window and the floating orb. Lives
in its own module so ui/floating_orb.py and ui/dot_grid.py can use it
without importing ui/app.py (which imports them — that would be circular).

Black background; the dot grid carries all the colour (see ui/dot_grid.py).
Text colours are checked against black: INK ~18:1, SLATE ~6.9:1.
"""

import tkinter.font as tkfont

BG = "#000000"
SURFACE = "#121416"  # composer and your message bubbles
LINE = "#24282B"  # composer border, disabled send button
LINE_HOVER = "#3A3F43"
INK = "#EDEFF0"
SLATE = "#8B969B"
ACCENT = "#22D3EE"  # send button, focus ring
ACCENT_HOVER = "#67E3F4"
SELECTION = "#1B4B55"

COLUMN_WIDTH = 580  # ~75 characters of body text

# Segoe UI Variable ships with Windows 11; plain Segoe UI is the Windows 10
# fallback. (GDI truncates long family names to 31 characters — hence
# "Display Semil".)
FONT_TEXT = ("Segoe UI Variable Text", "Segoe UI")
FONT_TEXT_SEMIBOLD = ("Segoe UI Variable Text Semibold", "Segoe UI Semibold")
FONT_SMALL = ("Segoe UI Variable Small", "Segoe UI")
FONT_DISPLAY_LIGHT = ("Segoe UI Variable Display Semil", "Segoe UI Semilight", "Segoe UI")

# Type scale, px
SIZE_SMALL = 12
SIZE_BODY = 15
SIZE_DISPLAY = 32


def pick_font(candidates: tuple[str, ...]) -> str:
    try:
        available = set(tkfont.families())
    except Exception:
        return candidates[-1]
    for name in candidates:
        if name in available:
            return name
    return candidates[-1]
