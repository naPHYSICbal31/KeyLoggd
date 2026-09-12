"""
charts.py

Canvas-drawn charts for the explainer view, in the app's dark theme.

Tk has no plotting library, so the marks are drawn directly. The specs follow
one house style rather than per-chart taste: thin marks capped at 24px with a
rounded data-end and a square baseline, a 2px surface gap between touching
bars, solid hairline grid one step off the surface, a legend whenever two
series share a plot, values direct-labelled selectively (the extreme, not
every mark), and axis/legend/value text in text tokens rather than the series
colour -- identity comes from the swatch beside the label.

Palette (chart-specific steps, validated against the #30302E panel surface for
the dark band, chroma floor, CVD separation, normal-vision separation and
contrast):

    SERIES_1  #D2724F   the product's terracotta, stepped into the dark
                        lightness band (the chrome accent #D97757 sits a
                        hair above its ceiling at L 0.672)
    SERIES_2  #3987e5   warm/cool counterpart -- the most CVD-robust pairing
                        available next to terracotta; worst all-pairs CVD
                        Delta E 22.4, normal-vision 28.0

The same two hues serve as the diverging poles (warm = above the enrolled
mean, cool = below) over a neutral grey midpoint, since a diverging midpoint
must read as "nothing".

Deliberate house deviation: numbers and axis ticks are set in the mono face and
titles in the serif face, because that is this product's type system.
"""

import tkinter as tk
from dataclasses import dataclass, field
from tkinter import font as tkfont

from keyloggd.ui.anim import Animator, blend, ease_out_cubic, local_progress
from keyloggd.ui.theme import BG_PANEL, FG_DIM, FG_LABEL, FG_TEXT, WARN, mono, ui

SERIES_1 = "#D2724F"
SERIES_2 = "#3987e5"
DIVERGE_POS = SERIES_1
DIVERGE_NEG = SERIES_2
NEUTRAL_MARK = "#54544F"      # de-emphasised marks (emphasis charts)

MAX_BAR_THICKNESS = 24
SURFACE_GAP = 2
DATA_END_RADIUS = 4


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

@dataclass
class Series:
    values: list
    label: str = ""
    colour: str = SERIES_1
    unit: str = ""


@dataclass
class Band:
    """A shaded region across a span of categories (e.g. the zero padding)."""
    start: float
    end: float
    label: str = ""


@dataclass
class RefLine:
    """A threshold or reference value. Dashed, because it is a threshold and
    not a gridline -- gridlines here are always solid hairlines."""
    value: float
    label: str = ""
    colour: str = WARN


@dataclass
class ChartSpec:
    kind: str = "columns"          # columns | diverging | hbars
    series: list = field(default_factory=list)
    categories: list = None        # tick labels, one per index
    x_label: str = ""
    y_label: str = ""
    bands: list = field(default_factory=list)
    reflines: list = field(default_factory=list)
    highlight: int = None          # index to emphasise (hbars)
    value_format: str = "{:.2f}"
    tick_format: str = "{:g}"
    hover_format: str = None       # e.g. "keystroke {i} - {s0} ms"
    annotate_max: bool = True      # direct-label the extreme of series 1
    annotate_index: int = None     # ...or this index, when the story is a
                                   # specific mark rather than the extreme
    annotate_label: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nice_ticks(lo, hi, count=4):
    """Round tick values covering [lo, hi] -- axis ticks carry the values that
    are not direct-labelled, so they need to read as clean numbers."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(count, 1)
    magnitude = 10.0 ** int(f"{raw:e}".split("e")[1])
    for factor in (1, 2, 2.5, 5, 10):
        step = factor * magnitude
        if raw <= step:
            break
    start = step * int(lo / step)
    ticks = []
    value = start
    # only ticks that actually fall inside the plot: one beyond the range
    # would draw its gridline and label outside the axes
    while value <= hi + step * 0.001:
        if value >= lo - step * 0.001:
            ticks.append(round(value, 10))
        value += step
    return ticks


def bar_points(x1, y1, x2, y2, radius, flip=False):
    """A bar with its data-end rounded and its baseline square.

    Vertical bars: y1 is the data end, y2 the baseline. Horizontal bars
    (flip=True): x2 is the data end, x1 the baseline. The rounding is signed,
    so a bar growing downward (a negative value on a diverging axis) or
    leftward rounds the correct end rather than bulging past it.
    """
    if not flip:
        radius = max(0.0, min(radius, abs(x2 - x1) / 2, abs(y2 - y1)))
        if radius < 1:
            return [x1, y1, x2, y1, x2, y2, x1, y2]
        step = radius if y2 > y1 else -radius
        return [
            x1, y2, x1, y1 + step, x1, y1, x1 + radius, y1,
            x2 - radius, y1, x2, y1, x2, y1 + step, x2, y2,
        ]
    radius = max(0.0, min(radius, abs(y2 - y1) / 2, abs(x2 - x1)))
    if radius < 1:
        return [x1, y1, x2, y1, x2, y2, x1, y2]
    step = radius if x2 > x1 else -radius
    return [
        x1, y1, x2 - step, y1, x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2, x2 - step, y2, x1, y2,
    ]


class Chart(tk.Canvas):
    """One plot. Call show(spec) to (re)draw; marks grow in with a stagger."""

    PAD_LEFT = 58
    PAD_RIGHT = 18
    PAD_TOP = 16
    PAD_BOTTOM = 38
    REVEAL_MS = 620

    def __init__(self, parent, height=260, surface=BG_PANEL):
        super().__init__(parent, height=height, bg=surface, bd=0,
                         highlightthickness=0)
        self.surface = surface
        self.spec = None
        self.reveal = 1.0
        self.hover_index = None
        self.anim = Animator(self)
        self._fonts = {}
        self.grid_colour = blend(surface, FG_DIM, 0.42)
        self.axis_colour = blend(surface, FG_DIM, 0.75)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)

    # -- public -------------------------------------------------------------

    def show(self, spec, *, animate=True):
        self.spec = spec
        self.hover_index = None
        if not animate:
            # a jump must win over a reveal still in flight, or the tween from
            # the phase we just left keeps writing and leaves this chart
            # frozen part-drawn
            self.anim.cancel("reveal")
            self.reveal = 1.0
            self._draw()
            return

        def frame(p):
            self.reveal = p
            self._draw()

        self.anim.tween(self.REVEAL_MS, frame, easing=ease_out_cubic,
                        key="reveal")

    def clear(self):
        self.anim.cancel("reveal")
        self.spec = None
        self.delete("all")

    # -- geometry -----------------------------------------------------------

    def _measure(self, text, font_spec):
        """Text width in pixels, so labels are placed rather than guessed."""
        key = tuple(font_spec)
        cached = self._fonts.get(key)
        if cached is None:
            cached = tkfont.Font(family=font_spec[0], size=font_spec[1],
                                 weight=font_spec[2])
            self._fonts[key] = cached
        return cached.measure(text)

    def _plot_box(self):
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        return (self.PAD_LEFT, self.PAD_TOP, w - self.PAD_RIGHT,
                h - self.PAD_BOTTOM)

    LABEL_CLEARANCE = 26   # px of air above the tallest bar: the direct
                           # label's height plus its gap

    def _value_range(self):
        """The axis range, scaled so the tallest bar leaves room for a label.

        Solved in pixels rather than as a magic multiplier, because how much
        headroom a given fraction buys depends on the plot height -- and on a
        diverging axis the bars only get half of it.
        """
        values = [v for s in self.spec.series for v in s.values]
        values += [r.value for r in self.spec.reflines]
        if not values:
            return 0.0, 1.0
        hi, lo = max(values), min(values)
        _, y0, _, y1 = self._plot_box()
        height = max(y1 - y0, 1)

        if self.spec.kind == "diverging":
            peak = max(abs(hi), abs(lo)) or 1.0
            # a diverging bar rises from the middle, so it has H/2 to play with
            room = 1.0 - min(0.6, 2.0 * self.LABEL_CLEARANCE / height)
            return -peak / room, peak / room

        if hi <= 0:
            return min(0.0, lo), 1.0
        room = 1.0 - min(0.6, self.LABEL_CLEARANCE / height)
        return min(0.0, lo), hi / room

    # -- drawing ------------------------------------------------------------

    def _draw(self):
        self.delete("all")
        if not self.spec:
            return
        if self.spec.kind == "hbars":
            self._draw_hbars()
        else:
            self._draw_columns()

    def _draw_axes(self, x0, y0, x1, y1, lo, hi):
        """Solid hairline grid one step off the surface, clean tick values."""
        for tick in nice_ticks(lo, hi):
            y = y1 - (tick - lo) / (hi - lo) * (y1 - y0)
            self.create_line(x0, y, x1, y, fill=self.grid_colour)
            self.create_text(x0 - 8, y, text=self.spec.tick_format.format(tick),
                             anchor="e", fill=FG_DIM, font=mono(9))
        self.create_line(x0, y1, x1, y1, fill=self.axis_colour)
        if self.spec.y_label:
            self.create_text(x0 - 46, y0 - 6, text=self.spec.y_label,
                             anchor="w", fill=FG_DIM, font=ui(9))
        if self.spec.x_label:
            self.create_text((x0 + x1) / 2, y1 + 26, text=self.spec.x_label,
                             fill=FG_DIM, font=ui(9))

    def _draw_bands(self, x0, y0, x1, y1, count):
        for band in self.spec.bands:
            step = (x1 - x0) / max(count, 1)
            bx0 = x0 + band.start * step
            bx1 = x0 + band.end * step
            self.create_rectangle(bx0, y0, bx1, y1,
                                  fill=blend(self.surface, FG_DIM, 0.13),
                                  outline="")
            if band.label:
                self.create_text((bx0 + bx1) / 2, y0 + 10, text=band.label,
                                 fill=FG_DIM, font=ui(9))

    def _draw_reflines(self, x0, y0, x1, y1, lo, hi):
        for ref in self.spec.reflines:
            y = y1 - (ref.value - lo) / (hi - lo) * (y1 - y0)
            # dashed *because* it is a threshold; grid stays solid
            self.create_line(x0, y, x1, y, fill=ref.colour, dash=(4, 3))
            if ref.label:
                self.create_text(x1, y - 8, text=ref.label, anchor="e",
                                 fill=ref.colour, font=ui(9))

    def _draw_legend(self, x0, y0, x1):
        """Always present for two or more series: identity never rests on
        colour matching alone."""
        if len(self.spec.series) < 2:
            return
        x = x1
        for series in reversed(self.spec.series):
            label = series.label or ""
            width = 16 + self._measure(label, ui(9))
            self.create_rectangle(x - width, y0 - 4, x - width + 10, y0 + 4,
                                  fill=series.colour, outline="")
            self.create_text(x - width + 16, y0, text=label, anchor="w",
                             fill=FG_LABEL, font=ui(9))
            x -= width + 14

    def _draw_columns(self):
        spec = self.spec
        x0, y0, x1, y1 = self._plot_box()
        lo, hi = self._value_range()
        count = max(len(s.values) for s in spec.series)
        self._draw_bands(x0, y0, x1, y1, count)
        self._draw_axes(x0, y0, x1, y1, lo, hi)

        band_w = (x1 - x0) / max(count, 1)
        series_count = len(spec.series)
        # cap thickness and let the leftover be air, minus the surface gap
        thickness = min(MAX_BAR_THICKNESS,
                        max(2.0, band_w / series_count - SURFACE_GAP))
        zero_y = y1 - (0 - lo) / (hi - lo) * (y1 - y0)

        if self.hover_index is not None and 0 <= self.hover_index < count:
            hx = x0 + self.hover_index * band_w
            self.create_rectangle(hx, y0, hx + band_w, y1,
                                  fill=blend(self.surface, FG_TEXT, 0.06),
                                  outline="")

        for s_index, series in enumerate(spec.series):
            for i, value in enumerate(series.values):
                grow = local_progress(self.reveal, i, count, spread=0.55)
                if grow <= 0:
                    continue
                span = (value - 0) / (hi - lo) * (y1 - y0) * grow
                left = (x0 + i * band_w
                        + (band_w - thickness * series_count
                           - SURFACE_GAP * (series_count - 1)) / 2
                        + s_index * (thickness + SURFACE_GAP))
                right = left + thickness
                colour = series.colour
                if spec.kind == "diverging":
                    colour = DIVERGE_POS if value >= 0 else DIVERGE_NEG
                data_end = zero_y - span
                self.create_polygon(
                    bar_points(left, data_end, right, zero_y,
                               DATA_END_RADIUS),
                    fill=colour, outline="", smooth=False)

        if spec.kind == "diverging":
            self.create_line(x0, zero_y, x1, zero_y, fill=self.axis_colour)

        self._draw_reflines(x0, y0, x1, y1, lo, hi)
        self._draw_category_ticks(x0, y1, band_w, count)
        self._draw_legend(x0, y0 - 6, x1)
        self._annotate_extreme(x0, y0, y1, band_w, count, lo, hi, zero_y)
        self._draw_readout(x0, y0, count)

    def _draw_category_ticks(self, x0, y1, band_w, count):
        """Sparse index ticks -- a label under every one of 32 bins is noise."""
        if not self.spec.categories:
            step = max(1, count // 8)
            labels = [(i, str(i)) for i in range(0, count, step)]
        else:
            step = max(1, len(self.spec.categories) // 10)
            labels = [(i, str(c)) for i, c in enumerate(self.spec.categories)
                      if i % step == 0]
        for i, text in labels:
            self.create_text(x0 + (i + 0.5) * band_w, y1 + 11, text=text,
                             fill=FG_DIM, font=mono(9))

    def _annotate_extreme(self, x0, y0, y1, band_w, count, lo, hi, zero_y):
        """One direct label: the extreme of the leading series."""
        spec = self.spec
        if not spec.annotate_max or self.reveal < 0.98 or not spec.series:
            return
        values = spec.series[0].values
        if not values:
            return
        if spec.annotate_index is not None and 0 <= spec.annotate_index < len(values):
            index = spec.annotate_index
        else:
            index = max(range(len(values)), key=lambda i: abs(values[i]))
        value = values[index]
        text = spec.annotate_label or spec.value_format.format(value)

        # measured, not clipped: pull the label inboard if it would overflow
        half = self._measure(text, mono(9)) / 2 + 4
        right_edge = self._plot_box()[2]
        x = min(max(x0 + (index + 0.5) * band_w, x0 + half), right_edge - half)

        # the label is usually wider than one band, so clear the tallest mark
        # anywhere beneath it -- a neighbouring series' bar is often the tall
        # one, and a label sitting on a bar is worse than no label at all
        first = max(0, int((x - half - x0) / band_w))
        last = min(count - 1, int((x + half - x0) / band_w))
        covered = [abs(v) for s in spec.series
                   for i, v in enumerate(s.values) if first <= i <= last]
        span = (max(covered) if covered else abs(value)) / (hi - lo) * (y1 - y0)
        y = zero_y - span - 10 if value >= 0 else zero_y + span + 12
        self.create_text(x, max(y, y0 + 6), text=text, fill=FG_TEXT,
                         font=mono(9))

    def _draw_readout(self, x0, y0, count):
        if self.hover_index is None or not (0 <= self.hover_index < count):
            return
        parts = []
        for series in self.spec.series:
            if self.hover_index < len(series.values):
                value = series.values[self.hover_index]
                parts.append(f"{series.label or 'value'} "
                             f"{self.spec.value_format.format(value)}"
                             f"{(' ' + series.unit) if series.unit else ''}")
        template = self.spec.hover_format or "{index}"
        head = template.format(index=self.hover_index)
        self.create_text(x0, y0 - 6, text="   ".join([head] + parts),
                         anchor="w", fill=FG_LABEL, font=mono(9))

    def _draw_hbars(self):
        """Magnitude by identity: one series, emphasis on the decisive row --
        never a value-ramp across nominal categories."""
        spec = self.spec
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        pad = 14
        label_w = 96
        series = spec.series[0]
        rows = list(enumerate(series.values))
        row_h = min(30, (h - pad * 2) // max(len(rows), 1))
        value_w = 52
        span = w - pad * 2 - label_w - value_w
        hi = max([abs(v) for v in series.values]
                 + [r.value for r in spec.reflines] + [1e-9]) * 1.05

        value_labels = []
        for i, value in rows:
            y = pad + i * row_h + row_h / 2
            grow = local_progress(self.reveal, i, len(rows), spread=0.5)
            emphasised = spec.highlight == i
            name = (spec.categories[i] if spec.categories
                    and i < len(spec.categories) else str(i))
            self.create_text(pad + label_w, y, text=name, anchor="e",
                             fill=blend(self.surface,
                                        FG_TEXT if emphasised else FG_LABEL,
                                        max(grow, 0.05)),
                             font=ui(11, bold=emphasised))
            thickness = min(MAX_BAR_THICKNESS, row_h - SURFACE_GAP * 3)
            length = max(1.0, min(span, value / hi * span * grow))
            bx = pad + label_w + 12
            self.create_polygon(
                bar_points(bx, y - thickness / 2, bx + length,
                           y + thickness / 2, DATA_END_RADIUS, flip=True),
                fill=SERIES_1 if emphasised else NEUTRAL_MARK, outline="",
                smooth=False)
            if grow > 0.9:
                # queued, not drawn yet: the threshold line goes on top of the
                # bars but must not cut through a value label
                value_labels.append((bx + length + 8, y,
                                     spec.value_format.format(value)))

        for ref in spec.reflines:
            x = pad + label_w + 12 + ref.value / hi * span
            fade = max(0.0, (self.reveal - 0.5) / 0.5)
            self.create_line(x, pad - 2, x, pad + len(rows) * row_h,
                             fill=blend(self.surface, ref.colour, fade),
                             dash=(4, 3))
            if ref.label:
                self.create_text(x, pad + len(rows) * row_h + 10,
                                 text=ref.label, anchor="n",
                                 fill=blend(self.surface, ref.colour, fade),
                                 font=ui(9))

        for x, y, text in value_labels:
            self.create_text(x, y, text=text, anchor="w", fill=FG_DIM,
                             font=mono(9))

    # -- hover --------------------------------------------------------------

    def _on_motion(self, event):
        if not self.spec or self.spec.kind == "hbars":
            return
        x0, y0, x1, y1 = self._plot_box()
        count = max((len(s.values) for s in self.spec.series), default=0)
        if not count or not (x0 <= event.x <= x1):
            return
        index = int((event.x - x0) / ((x1 - x0) / count))
        index = max(0, min(count - 1, index))
        if index != self.hover_index:
            self.hover_index = index
            self._draw()

    def _on_leave(self, _event=None):
        if self.hover_index is not None:
            self.hover_index = None
            self._draw()

