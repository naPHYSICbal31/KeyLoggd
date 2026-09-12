"""
theme.py

Shared Claude-inspired dark theme for the project's Tkinter frontends
(the app, and the typing test's palette which this mirrors).

Palette: warm dark charcoal background, cream text, terracotta accent.

Fonts are *resolved at runtime* against the families Tk can actually see, so
the UI looks intentional whether or not the Anthropic Serif / Roboto Mono
families are installed on the machine (Tk silently falls back to a generic
default otherwise, which reads as a bug rather than a fallback).
Call init_fonts(root) once, after creating the Tk root.
"""

import tkinter as tk
from tkinter import font as tkfont

from keyloggd.ui.anim import Animator, blend, ease_out_cubic

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG = "#262624"            # main window background (warm dark charcoal)
BG_PANEL = "#30302E"      # slightly lighter panel background
BG_INPUT = "#1F1F1E"      # sunken field background
BG_HOVER = "#3A3A37"      # hover state for clickable pills
FG_TEXT = "#F0EEE6"       # cream body text
FG_DIM = "#6B6A66"        # dim gray: not-yet-typed text, inactive tabs
FG_LABEL = "#8A8985"      # brighter dim, for legible secondary labels
FG_CORRECT = "#F0EEE6"    # correctly typed characters
FG_INCORRECT = "#E0685A"  # incorrect characters (soft red)
ACCENT = "#D97757"        # Claude-esque terracotta accent
ACCENT_DIM = "#B85F42"    # darker terracotta for hover/borders
OK = "#7FA87F"            # muted green for success states
WARN = "#D9A257"          # amber for warnings / low-confidence

# Font family preference lists, best first.
_UI_REGULAR_PREFS = [
    "Anthropic Serif Text Light", "Anthropic Serif Text", "Anthropic Serif",
    "Georgia", "Cambria", "Times New Roman",
]
_UI_BOLD_PREFS = [
    "Anthropic Serif Text Bold", "Anthropic Serif Text", "Anthropic Serif",
    "Georgia", "Cambria", "Times New Roman",
]
_MONO_PREFS = [
    "Roboto Mono", "JetBrains Mono", "Cascadia Mono", "Consolas",
    "DejaVu Sans Mono", "Courier New",
]

# Filled in by init_fonts(); the pre-init values are only ever used if a
# caller forgets to call it (in which case Tk's own defaults apply anyway).
UI_REGULAR = "TkDefaultFont"
UI_BOLD = "TkDefaultFont"
MONO = "TkFixedFont"

# "Anthropic Serif Text Bold" is its own family rather than a bold flag on the
# regular family, so when that family is available we must NOT also ask Tk for
# weight="bold" (it would synthesize a second, uglier bolding on top).
_BOLD_IS_OWN_FAMILY = False


def _first_available(prefs, families, fallback):
    lower = {f.lower(): f for f in families}
    for want in prefs:
        if want.lower() in lower:
            return lower[want.lower()]
    return fallback


def init_fonts(root: tk.Misc):
    """Resolve the font families against what Tk can see. Call once, after
    the Tk root exists."""
    global UI_REGULAR, UI_BOLD, MONO, _BOLD_IS_OWN_FAMILY
    families = tkfont.families(root)
    UI_REGULAR = _first_available(_UI_REGULAR_PREFS, families, "TkDefaultFont")
    UI_BOLD = _first_available(_UI_BOLD_PREFS, families, "TkDefaultFont")
    MONO = _first_available(_MONO_PREFS, families, "TkFixedFont")
    _BOLD_IS_OWN_FAMILY = "bold" in UI_BOLD.lower()


def ui(size, bold=False):
    """A UI-chrome font spec tuple, e.g. ui(14, bold=True)."""
    if bold:
        weight = "normal" if _BOLD_IS_OWN_FAMILY else "bold"
        return (UI_BOLD, size, weight)
    return (UI_REGULAR, size, "normal")


def mono(size, bold=False):
    """A fixed-width font spec tuple, for typing text and numbers."""
    return (MONO, size, "bold" if bold else "normal")


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class RoundedPanel(tk.Canvas):
    """A rounded-rectangle 'pill' panel. Children go into `.inner`.

    Arcs at the corners plus two
    overlapping rectangles, drawn on a canvas that auto-sizes to its inner
    frame's requested size.
    """

    def __init__(self, parent, fill=BG_PANEL, radius=10, padding=8, bg=BG):
        super().__init__(parent, bg=bg, bd=0, highlightthickness=0)
        self.fill = fill
        self.radius = radius
        self.padding = padding
        self.inner = tk.Frame(self, bg=fill)
        self.create_window(padding, padding, anchor="nw", window=self.inner)
        self.bind("<Configure>", self._draw)
        self.inner.bind("<Configure>", self._resize)

    def _resize(self, event=None):
        self.configure(
            width=self.inner.winfo_reqwidth() + self.padding * 2,
            height=self.inner.winfo_reqheight() + self.padding * 2,
        )
        self._draw()

    def set_fill(self, fill):
        self.fill = fill
        self.inner.configure(bg=fill)
        self._draw()

    def _draw(self, event=None):
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        radius = min(self.radius, width // 2, height // 2)
        self.delete("panel")
        self.create_arc(0, 0, radius * 2, radius * 2, start=90, extent=90,
                        fill=self.fill, outline=self.fill, tags="panel")
        self.create_arc(width - radius * 2, 0, width, radius * 2, start=0,
                        extent=90, fill=self.fill, outline=self.fill, tags="panel")
        self.create_arc(0, height - radius * 2, radius * 2, height, start=180,
                        extent=90, fill=self.fill, outline=self.fill, tags="panel")
        self.create_arc(width - radius * 2, height - radius * 2, width, height,
                        start=270, extent=90, fill=self.fill, outline=self.fill,
                        tags="panel")
        self.create_rectangle(radius, 0, width - radius, height,
                              fill=self.fill, outline=self.fill, tags="panel")
        self.create_rectangle(0, radius, width, height - radius,
                              fill=self.fill, outline=self.fill, tags="panel")
        self.tag_lower("panel")


def rounded_rect_points(x1, y1, x2, y2, radius):
    """Control points for a rounded rectangle drawn as a smoothed polygon.

    Doubling each corner point is what turns Tk's spline into a corner arc:
    the two straight-edge endpoints pull the curve flat along the sides, and
    the corner itself acts as the control point between them.
    """
    return [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]


class PillButton(tk.Canvas):
    """A rounded, Claude-styled button.

    Tk has no rounded corners and no border radius, so the button is drawn:
    a smoothed polygon for the body plus a canvas text item for the label,
    on a canvas whose own background matches the surface behind it, so the
    corners read as rounded rather than as notches.

        kind="primary"  filled terracotta, dark label -- the main action
        kind="ghost"    panel fill with a hairline border, cream label

    Hover, press and disable all animate the body and border colours (see
    anim.Animator), which is what makes a drawn button feel like a control
    rather than a picture of one.
    """

    RADIUS = 9
    BORDER = 1

    def __init__(self, parent, text, command, kind="ghost", size=11, bg=BG,
                 padx=16, pady=9, radius=RADIUS):
        self.kind = kind
        self.command = command
        self.surface = bg
        self.radius = radius
        self._enabled = True

        self.font = ui(size, bold=(kind == "primary"))
        metrics = tkfont.Font(family=self.font[0], size=self.font[1],
                              weight=self.font[2])
        width = metrics.measure(text) + padx * 2
        height = metrics.metrics("linespace") + pady * 2

        super().__init__(parent, width=width, height=height, bg=bg, bd=0,
                         highlightthickness=0, cursor="hand2")

        self.fill, self.border, self.label = self._palette(kind, bg)
        self.hover_fill, self.hover_border = self._hover_palette(kind, bg)

        self._fill = self.fill
        self._border = self.border
        self._label = self.label
        self._anim = Animator(self)

        self._shape_id = None
        self._text_id = None
        self.text = text
        self._draw()

        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    # -- palettes -----------------------------------------------------------

    @staticmethod
    def _palette(kind, surface):
        """(body, border, label) colours at rest."""
        if kind == "primary":
            return ACCENT, blend(ACCENT, "#000000", 0.18), BG
        # ghost: a body lifted clearly off the surface it sits on, so the
        # button reads as a raised control rather than as outlined text, with
        # a border warmed towards the accent instead of a hard grey outline
        return (blend(surface, BG_HOVER, 0.7),
                blend(surface, ACCENT_DIM, 0.6), FG_TEXT)

    @staticmethod
    def _hover_palette(kind, surface):
        if kind == "primary":
            return blend(ACCENT, "#FFFFFF", 0.12), ACCENT
        return blend(BG_HOVER, FG_LABEL, 0.1), ACCENT

    # -- drawing ------------------------------------------------------------

    def _draw(self):
        width = max(self.winfo_reqwidth(), self.winfo_width())
        height = max(self.winfo_reqheight(), self.winfo_height())
        inset = self.BORDER
        points = rounded_rect_points(inset, inset, width - inset - 1,
                                     height - inset - 1,
                                     min(self.radius, height // 2))

        if self._shape_id is None:
            self._shape_id = self.create_polygon(
                points, smooth=True, splinesteps=32, fill=self._fill,
                outline=self._border, width=self.BORDER)
            self._text_id = self.create_text(
                width / 2, height / 2, text=self.text, fill=self._label,
                font=self.font)
        else:
            self.coords(self._shape_id, points)
            self.coords(self._text_id, width / 2, height / 2)

    def _repaint(self):
        if self._shape_id is None:
            return
        self.itemconfigure(self._shape_id, fill=self._fill,
                           outline=self._border)
        self.itemconfigure(self._text_id, fill=self._label)

    def _tween_colours(self, fill, border, label, duration=140, key="hover"):
        start = (self._fill, self._border, self._label)

        def frame(p):
            self._fill = blend(start[0], fill, p)
            self._border = blend(start[1], border, p)
            self._label = blend(start[2], label, p)
            self._repaint()

        self._anim.tween(duration, frame, easing=ease_out_cubic, key=key)

    # -- interaction --------------------------------------------------------

    def _on_click(self, _event=None):
        if not self._enabled:
            return
        # press feedback lands before the command runs, so a button that
        # kicks off slow work still acknowledges the click immediately
        pressed = blend(self._fill, "#000000", 0.22)
        self._anim.cancel("hover")
        self._fill = pressed
        self._repaint()
        if self.command:
            self.after_idle(self.command)

    def _on_release(self, _event=None):
        if self._enabled:
            self._tween_colours(self.hover_fill, self.hover_border, self.label,
                                duration=110)

    def _on_enter(self, _event=None):
        if self._enabled:
            self._tween_colours(self.hover_fill, self.hover_border, self.label)

    def _on_leave(self, _event=None):
        if self._enabled:
            self._tween_colours(self.fill, self.border, self.label)

    def set_text(self, text):
        """Relabel in place (a play/pause toggle), resizing to fit."""
        self.text = text
        metrics = tkfont.Font(family=self.font[0], size=self.font[1],
                              weight=self.font[2])
        pad = self.winfo_reqwidth() - metrics.measure(self.itemcget(
            self._text_id, "text"))
        self.configure(width=metrics.measure(text) + pad)
        self.itemconfigure(self._text_id, text=text)
        self._draw()

    def set_enabled(self, enabled):
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        if enabled:
            self._tween_colours(self.fill, self.border, self.label,
                                duration=180, key="enable")
        else:
            # a disabled button keeps its shape but drops back to the surface,
            # so it reads as unavailable rather than merely dim
            self._tween_colours(blend(self.surface, BG_PANEL, 0.6),
                                blend(self.surface, FG_DIM, 0.35), FG_DIM,
                                duration=180, key="enable")


class Spinner(tk.Canvas):
    """A rotating arc shown while a background job runs.

    Sized to sit inline with a text label; call start()/stop(). It packs and
    unpacks itself so the layout closes up when there is nothing to wait for.
    """

    def __init__(self, parent, size=16, bg=BG, colour=ACCENT, width=2):
        super().__init__(parent, width=size, height=size, bg=bg, bd=0,
                         highlightthickness=0)
        self.size = size
        self.colour = colour
        self.arc_width = width
        self._anim = Animator(self)
        self._pack_options = None
        self._angle = 0

    def start(self, **pack_options):
        if self._pack_options is None:
            self._pack_options = pack_options or {"side": "left"}
            self.pack(**self._pack_options)
        self._anim.repeat(900, self._frame, key="spin")

    def stop(self):
        self._anim.cancel("spin")
        self.delete("all")
        if self._pack_options is not None:
            self.pack_forget()
            self._pack_options = None

    def _frame(self, p):
        inset = self.arc_width
        self.delete("all")
        # the extent breathes as it spins, which reads as motion even at the
        # moments the arc's ends are hidden behind the rotation
        extent = 80 + 60 * abs(0.5 - p) * 2
        self.create_arc(inset, inset, self.size - inset, self.size - inset,
                        start=-p * 360, extent=-extent, style="arc",
                        outline=self.colour, width=self.arc_width)
