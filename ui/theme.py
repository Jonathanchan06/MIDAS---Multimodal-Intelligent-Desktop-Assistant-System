"""Shared palette for the main window and the floating orb. Lives in its
own module so ui/orb.py can use it without importing ui/app.py (which
imports ui/orb.py — that would be circular)."""

BG = "#ffffff"
TEXT_PRIMARY = "#14181a"
TEXT_MUTED = "#a7b0b2"
TEXT_FAINT = "#c1c8ca"
TEXT_USER = "#6b7578"
ACCENT = "#0ba5c2"
BORDER = "#e7ebec"
BORDER_STRONG = "#e5e9ea"
RING_DASHED = "#dbe2e3"
RING_INNER = "#cfe9ee"
CORE_FILL = "#f3f8f9"
