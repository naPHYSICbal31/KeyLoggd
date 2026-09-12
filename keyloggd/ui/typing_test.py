"""
typing_test.py

The Monkeytype-style typing test -- and, in this project, the enrollment
input. Someone setting up a profile types here rather than repeating a short
phrase twelve times: the run is longer, the words are unpredictable, and the
rhythm it captures is the one they actually type with.

Every keydown and keyup on the typing area is timestamped with
time.perf_counter(), and a finished run is cut into fixed-width samples by
pipeline.samples.split_into_samples -- so one 30-second test yields several
enrollment samples in the exact JSON schema the rest of the pipeline reads.

Two ways to run it:

    embedded    TypingTest(parent, on_finish=...) inside another view. The
                host routes keystrokes in through handle_keypress /
                handle_keyrelease and receives finished samples through
                on_finish. This is what the enroll view uses.
    standalone  python -m keyloggd.ui.typing_test -- the plain typing test,
                with an export button instead of a host to hand samples to.

Finishing a test swaps the typing area for a results screen: the headline
stats beside a graph of net wpm, per-second raw wpm, and the seconds that
contained a mistake.
"""

import json
import math
import random
import time
import tkinter as tk
import tkinter.simpledialog as simpledialog
from tkinter import filedialog, messagebox

from keyloggd.pipeline.samples import MIN_KEYS, split_into_samples
from keyloggd.ui.anim import Animator, blend
from keyloggd.ui.theme import (ACCENT, ACCENT_DIM, BG, BG_PANEL, FG_CORRECT,
                               FG_DIM, FG_INCORRECT, FG_LABEL, FG_TEXT,
                               RoundedPanel, init_fonts, mono, ui)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CARET_X_OFFSET = -21      # pixels from Tk's character bounding-box edge
CARET_Y_OFFSET = 0        # pixels from the top of Tk's character bounding box
CARET_WIDTH = 3           # caret thickness in pixels
CARET_ANIMATION_STEPS = 12
CARET_ANIMATION_DELAY_MS = 12
CARET_BLINK_STEPS = 12
CARET_BLINK_DELAY_MS = 45
TYPE_FONT_SIZE = 26       # sized to fit the three-line typing container
DISPLAY_LINES = 3

#: Recorded under this name in the enrollment file. The typing test deals
#: fresh words every run, so no single phrase describes what was typed --
#: and nothing downstream reads the phrase, which is the point: the
#: features are timing, not text.
FREE_TEXT_LABEL = "free text (typing test)"

WORDLIST = (
    "the of and to in is you that it he was for on are as with his they "
    "at be this have from or one had by word but not what all were we "
    "when your can said there use each which she do how their if will "
    "up other about out many then them these so some her would make "
    "like him into time has look two more write go see number no way "
    "could people my than first water been call who oil its now find "
    "long down day did get come made may part over new sound take only "
    "little work know place year live me back give most very after "
    "thing our just name good sentence man think say great where help "
    "through much before line right too mean old any same tell boy "
    "follow came want show also around form three small set put end "
    "does another well large must big even such because turn here why "
    "ask went men read need land different home us move try kind hand "
).split()

# Mode option values
TEST_DURATIONS = [15, 30, 60, 120]
WORD_COUNTS = [10, 25, 50, 100]
QUOTE_LENGTHS = ["short", "medium", "long"]

MODES = [
    ("time", "\u23F1"),    # clock
    ("words", "A"),        # letter
    ("quote", "\u275D"),   # quote mark
    ("zen", "\u25B3"),     # triangle
    ("custom", "\u2699"),  # gear/wrench-ish
]

# Original filler quotes (not sourced from any copyrighted work), grouped by length.
QUOTES = {
    "short": [
        "The quick brown fox jumps over the lazy dog.",
        "Practice makes perfect when you type every day.",
        "Simple words build strong typing habits fast.",
        "Small steady steps add up to real speed.",
    ],
    "medium": [
        "Typing quickly is a skill built through steady practice, patience, "
        "and a willingness to make mistakes along the way.",
        "Every keystroke you make is a small step toward faster, more "
        "accurate typing, so keep your fingers moving and stay focused.",
        "A calm mind and relaxed hands often type faster than a tense one "
        "rushing to finish the sentence.",
        "Good posture and a steady rhythm matter more than raw effort when "
        "you are trying to build lasting speed.",
    ],
    "long": [
        "Learning to type well is less about raw speed and more about "
        "rhythm, consistency, and the quiet confidence that comes from "
        "thousands of repeated motions until they finally feel natural "
        "under your fingers.",
        "The best typists rarely look at their keyboards at all, relying "
        "instead on muscle memory built over months of deliberate "
        "practice, short daily sessions, and an ongoing curiosity about "
        "how to shave a few more words off their per minute average.",
        "There is a particular kind of satisfaction in watching your "
        "words per minute climb week after week, not because the number "
        "itself matters so much, but because it reflects real, "
        "measurable progress earned through patience and steady effort.",
    ],
}

DEFAULT_CUSTOM_TEXT = (
    "This is custom mode. Click the custom tab again to replace this "
    "text with anything you would like to practice typing."
)


# ---------------------------------------------------------------------------
# Results graph
# ---------------------------------------------------------------------------

class WpmChart(tk.Canvas):
    """The Monkeytype-style results graph, drawn straight onto a Tk canvas.

    Three marks share a single words-per-minute scale, so there is no second
    axis: the settled line is net wpm as it converges over the run, the jagged
    line is the raw wpm of each individual second, and an X sits on the raw
    line wherever that second contained a mistake. The X is a different shape,
    not merely a different colour, so the error marks survive a colourblind or
    greyscale read.

    The two series colours are the chart steps checked against this panel
    surface (the product terracotta stepped into the dark lightness band, plus
    its warm/cool counterpart); numbers and labels stay in text tokens, with
    identity carried by the swatch beside each legend label.
    """

    WPM_COLOUR = "#D2724F"
    RAW_COLOUR = "#3987E5"
    ERR_COLOUR = FG_INCORRECT
    GRID = "#3A3A37"

    PAD_L, PAD_R, PAD_T, PAD_B = 54, 22, 38, 34
    LINE_WIDTH = 2

    def __init__(self, parent, height=250):
        super().__init__(parent, bg=BG_PANEL, bd=0, highlightthickness=0,
                         height=height)
        self.samples = []
        self._geom = None
        self._hover = None
        self.tick_font = mono(9)
        self.label_font = ui(11)
        self.value_font = mono(11)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)

    def show(self, samples):
        self.samples = list(samples)
        self._hover = None
        self._draw()

    def clear(self):
        self.samples = []
        self._hover = None
        self.delete("all")

    # -- scales ---------------------------------------------------------
    @staticmethod
    def _axis_steps(peak, count=5):
        """A round gridline step, and the smallest multiple of it above `peak`.

        The ceiling is snapped to the step rather than fixed at `count` steps,
        so a 108 wpm peak tops out at 125 instead of stranding the data in the
        bottom half of the plot.
        """
        peak = max(peak, 1.0)
        step = 1
        for candidate in (1, 2, 4, 5, 10, 20, 25, 40, 50, 100, 200, 250, 400, 500, 1000):
            step = candidate
            if candidate >= peak / count:
                break
        return math.ceil(peak / step) * step, step

    @staticmethod
    def _time_step(span):
        for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
            if span / step <= 6:
                return step
        return 600

    def _px(self, t):
        x0, _y0, x1, _y1, span, _top = self._geom
        return x0 + (t / span if span else 0.0) * (x1 - x0)

    def _py(self, value):
        _x0, y0, _x1, y1, _span, top = self._geom
        return y1 - (value / top if top else 0.0) * (y1 - y0)

    # -- drawing --------------------------------------------------------
    def _draw(self):
        self.delete("all")
        self._geom = None
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:
            return
        if len(self.samples) < 2:
            self.create_text(width / 2, height / 2, text="too short to graph",
                             fill=FG_DIM, font=self.label_font)
            return

        x0, y0 = self.PAD_L, self.PAD_T
        x1, y1 = width - self.PAD_R, height - self.PAD_B
        if x1 - x0 < 60 or y1 - y0 < 50:
            return

        span = max(s["t"] for s in self.samples) or 1.0
        peak = max(max(s["wpm"], s["raw"]) for s in self.samples)
        top, step = self._axis_steps(peak)
        self._geom = (x0, y0, x1, y1, span, top)

        self._draw_grid(x0, x1, y1, top, step, span)
        self._draw_series()
        self._draw_errors()
        self._draw_end_label()
        self._draw_legend(x0)

    def _draw_grid(self, x0, x1, y1, top, step, span):
        value = 0
        while value <= top + 1e-9:
            y = self._py(value)
            self.create_line(x0, y, x1, y, fill=self.GRID)
            self.create_text(x0 - 10, y, text=str(int(value)), anchor="e",
                             fill=FG_DIM, font=self.tick_font)
            value += step

        t_step = self._time_step(span)
        mark = t_step
        while mark < span - t_step * 0.4:
            self.create_text(self._px(mark), y1 + 8, text=f"{mark:g}",
                             anchor="n", fill=FG_DIM, font=self.tick_font)
            mark += t_step
        self.create_text(x1, y1 + 8, text=f"{span:.0f}s", anchor="ne",
                         fill=FG_DIM, font=self.tick_font)

    def _draw_series(self):
        raw_points = []
        wpm_points = []
        for sample in self.samples:
            x = self._px(sample["t"])
            raw_points.extend((x, self._py(sample["raw"])))
            wpm_points.extend((x, self._py(sample["wpm"])))
        for points, colour in ((raw_points, self.RAW_COLOUR),
                               (wpm_points, self.WPM_COLOUR)):
            self.create_line(*points, fill=colour, width=self.LINE_WIDTH,
                             smooth=True, splinesteps=24,
                             capstyle="round", joinstyle="round")

    def _draw_errors(self):
        for sample in self.samples:
            if sample["errors"] <= 0:
                continue
            x = self._px(sample["t"])
            y = self._py(sample["raw"])
            # a surface-coloured pass first, so the glyph keeps a gap from
            # whichever line it happens to land on
            for colour, weight in ((BG_PANEL, self.LINE_WIDTH + 4),
                                   (self.ERR_COLOUR, 2)):
                self.create_line(x - 4, y - 4, x + 4, y + 4,
                                 fill=colour, width=weight, capstyle="round")
                self.create_line(x - 4, y + 4, x + 4, y - 4,
                                 fill=colour, width=weight, capstyle="round")

    def _draw_end_label(self):
        _x0, y0, _x1, _y1, _span, _top = self._geom
        last = self.samples[-1]
        x = self._px(last["t"])
        y = max(y0 + 8, self._py(last["wpm"]) - 12)
        self.create_text(x - 4, y, text=f"{last['wpm']:.0f}", anchor="se",
                         fill=FG_TEXT, font=self.value_font)

    def _draw_legend(self, x0):
        y = self.PAD_T / 2
        x = x0
        entries = ((self.WPM_COLOUR, "wpm", False),
                   (self.RAW_COLOUR, "raw", False),
                   (self.ERR_COLOUR, "errors", True))
        for colour, label, is_cross in entries:
            if is_cross:
                self.create_line(x + 2, y - 4, x + 12, y + 4, fill=colour, width=2)
                self.create_line(x + 2, y + 4, x + 12, y - 4, fill=colour, width=2)
            else:
                self.create_line(x, y, x + 14, y, fill=colour,
                                 width=self.LINE_WIDTH, capstyle="round")
            item = self.create_text(x + 21, y, text=label, anchor="w",
                                    fill=FG_LABEL, font=self.label_font)
            x = self.bbox(item)[2] + 18

    # -- hover ----------------------------------------------------------
    def _on_motion(self, event):
        if self._geom is None:
            return
        x0, y0, x1, y1, span, _top = self._geom
        if not (x0 - 8 <= event.x <= x1 + 8 and y0 - 12 <= event.y <= y1 + 12):
            self._on_leave()
            return
        t = (event.x - x0) / max(1.0, float(x1 - x0)) * span
        index = min(range(len(self.samples)),
                    key=lambda i: abs(self.samples[i]["t"] - t))
        if index == self._hover:
            return
        self._hover = index
        self._draw_hover()

    def _on_leave(self, _event=None):
        if self._hover is None:
            return
        self._hover = None
        self.delete("hover")

    def _draw_hover(self):
        self.delete("hover")
        if self._hover is None or self._geom is None:
            return
        sample = self.samples[self._hover]
        _x0, y0, x1, y1, _span, _top = self._geom
        x = self._px(sample["t"])
        self.create_line(x, y0, x, y1, fill=FG_DIM, tags="hover")
        for value, colour in ((sample["raw"], self.RAW_COLOUR),
                              (sample["wpm"], self.WPM_COLOUR)):
            y = self._py(value)
            self.create_oval(x - 4, y - 4, x + 4, y + 4, fill=colour,
                             outline=BG_PANEL, width=2, tags="hover")
        parts = [f"{sample['t']:.0f}s", f"{sample['wpm']:.0f} wpm",
                 f"{sample['raw']:.0f} raw"]
        if sample["errors"]:
            plural = "" if sample["errors"] == 1 else "s"
            parts.append(f"{sample['errors']} error{plural}")
        self.create_text(x1, self.PAD_T / 2, text="   ".join(parts), anchor="e",
                         fill=FG_TEXT, font=self.value_font, tags="hover")


# ---------------------------------------------------------------------------
# The typing test
# ---------------------------------------------------------------------------

class TypingTest(tk.Frame):
    """A Monkeytype-style typing test that doubles as a keystroke recorder.

    Embedding contract (mirrors the phrase capture widget, so a host can hold
    either one behind the same code):

        active                   set False and the widget ignores keystrokes
        handle_keypress(event)   feed it a <KeyPress>; returns "break" when
        handle_keyrelease(event) the key must not travel on (Tab, which would
                                 otherwise move focus out of the test)
        reset()                  abandon what is on screen, deal a new test
        appear(delay=...)        fade in, for a host that animates its views
        on_typing_start()        fires on the first keystroke of a run, so a
                                 host can settle any animation that would sit
                                 between a key and the timestamp it is given

    on_finish(samples, summary) fires once a run ends, with the run already
    cut into pipeline-schema samples. `summary` carries the run's headline
    numbers (wpm, accuracy, characters, seconds). A host that wants to say
    what it did with the samples can call set_result_note().
    """

    HINT = "tab: new test   |   esc: stop   |   type to begin"

    def __init__(self, parent, on_finish=None, on_typing_start=None,
                 embedded=False, bg=BG):
        super().__init__(parent, bg=bg)
        self.on_finish = on_finish
        self.on_typing_start = on_typing_start
        self.embedded = embedded
        self.active = True

        # --- mode / option state ------------------------------------------
        self.mode = "time"                # time | words | quote | zen | custom
        self.test_duration = 30           # seconds, for time mode
        self.word_count = 25              # words, for words mode
        self.quote_length = "medium"      # short | medium | long, for quotes
        self.punctuation = False
        self.numbers = False
        self.custom_text = ""             # set via the custom-mode dialog

        self.state = "idle"  # idle -> running -> finished
        self.words = []
        self.full_text = ""
        self.pos = 0            # index of next character to type
        self.typed_correct = 0
        self.typed_incorrect = 0
        self.start_perf = None
        self.timer_job = None
        self.caret_job = None
        self.caret_blink_job = None
        self.caret_retry_job = None
        self.caret_visible = True
        self.caret_blink_level = 0
        self.caret_blink_direction = 1
        self.caret_position = None
        self.display_scroll_line = 0

        # per-second samples behind the results graph:
        # {"t", "wpm" (net, cumulative), "raw" (that second alone), "errors"}
        self.wpm_history = []
        self._last_sample_t = 0.0
        self._last_sample_typed = 0
        self._last_sample_incorrect = 0

        # raw keystroke capture, in the same schema as the rest of the project
        self.raw_events = []      # list of {"key","type","t"}
        self.session_perf0 = None
        self.held_keys = set()    # keys currently down, to drop auto-repeat
        self.anim = Animator(self)

        self._build_ui()
        self._rebuild_options()
        self._new_test()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        bg = self.cget("bg")

        # A host supplies its own title bar, so the wordmark -- and the space
        # it costs -- is only spent when this is the whole window.
        if not self.embedded:
            top = tk.Frame(self, bg=bg)
            top.place(relx=0, rely=0, relwidth=1, x=24, y=18, width=-48,
                      height=30)
            tk.Label(top, text="typetest", fg=ACCENT, bg=bg,
                     font=ui(18, bold=True)).pack(side="left")

        # controls: toggles pill + mode pill + options pill, centred as a group
        self.controls_rely = 0.0 if self.embedded else 0.14
        controls = tk.Frame(self, bg=bg)
        self.controls_frame = controls
        controls.place(relx=0.5, rely=self.controls_rely, anchor="n")

        toggles_panel = RoundedPanel(controls, BG_PANEL, radius=6, bg=bg)
        toggles_panel.pack(side="left")
        self.punct_label = tk.Label(toggles_panel.inner, text="@ punctuation",
                                    fg=FG_DIM, bg=BG_PANEL, font=ui(12),
                                    cursor="hand2", padx=8)
        self.punct_label.pack(side="left")
        self.punct_label.bind("<Button-1>",
                              lambda e: self._toggle_punctuation())

        self.numbers_label = tk.Label(toggles_panel.inner, text="# numbers",
                                      fg=FG_DIM, bg=BG_PANEL, font=ui(12),
                                      cursor="hand2", padx=8)
        self.numbers_label.pack(side="left")
        self.numbers_label.bind("<Button-1>", lambda e: self._toggle_numbers())

        mode_panel = RoundedPanel(controls, BG_PANEL, radius=6, bg=bg)
        mode_panel.pack(side="left", padx=24)
        self.mode_labels = {}
        for name, icon in MODES:
            lbl = tk.Label(mode_panel.inner, text=f"{icon} {name}", fg=FG_DIM,
                           bg=BG_PANEL, font=ui(12), cursor="hand2", padx=8)
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e, nm=name: self._on_mode_click(nm))
            self.mode_labels[name] = lbl

        options_panel = RoundedPanel(controls, BG_PANEL, radius=6, bg=bg)
        options_panel.pack(side="left")
        self.options_frame = options_panel.inner

        # main typing area
        mid = tk.Frame(self, bg=bg)
        self.mid_place = (
            {"relx": 0, "rely": 0.11, "relwidth": 1, "relheight": 0.82,
             "x": 12, "width": -24}
            if self.embedded else
            {"relx": 0, "rely": 0.2, "relwidth": 1, "relheight": 0.7,
             "x": 24, "width": -48})
        mid.place(**self.mid_place)
        controls.lift()

        # Status readout (countdown / word progress / elapsed) sits at the
        # top-left of the text container, not the window's top-right corner.
        # Embedded, the container is shorter, so the countdown sits higher and
        # the words lower -- at the standalone spacing the two would overlap.
        self.status_rely = 0.22 if self.embedded else 0.38
        self.text_rely = 0.58 if self.embedded else 0.5
        self.status_row = status_row = tk.Frame(mid, bg=bg)
        status_row.place(relx=.015, rely=self.status_rely, anchor="sw")
        self.timer_label = tk.Label(status_row, text=str(self.test_duration),
                                    fg=ACCENT, bg=bg, font=ui(32, bold=True))
        self.timer_label.pack(side="left")

        self.text = tk.Text(
            mid, wrap="word", bg=bg, fg=FG_DIM, insertbackground=ACCENT,
            font=mono(TYPE_FONT_SIZE), height=DISPLAY_LINES, borderwidth=0,
            highlightthickness=0, padx=20, pady=0, cursor="arrow",
            spacing1=0, spacing3=0,
        )
        self.text.place(relx=0.5, rely=self.text_rely, anchor="center",
                        relwidth=1.0)
        status_row.lift()
        self.text.tag_configure("correct", foreground=FG_CORRECT)
        self.text.tag_configure("incorrect", foreground=FG_INCORRECT,
                                underline=True)
        self.text.tag_configure("untyped", foreground=FG_DIM)
        self.text.config(state="disabled")
        self.caret = tk.Frame(self.text, bg=ACCENT, width=CARET_WIDTH, bd=0,
                              highlightthickness=0)

        self._build_results()

        # Bottom hint bar. A host has its own footer and would print the
        # same keys twice, so embedded runs without one.
        if self.embedded:
            self.bottom_frame = None
            self.bottom_place = None
        else:
            self.bottom_place = {"relx": 0, "rely": 1, "anchor": "sw",
                                 "relwidth": 1, "x": 24, "width": -48,
                                 "y": -16}
            self.bottom_frame = tk.Frame(self, bg=bg)
            self.bottom_frame.place(**self.bottom_place)
            tk.Label(self.bottom_frame, text=self.HINT, fg=FG_DIM, bg=bg,
                     font=ui(12)).pack(side="left")

        # Moving the mouse means the typist has stopped typing, so the chrome
        # that hid itself mid-run comes back. bind_all, because the pointer is
        # nearly always over a child widget rather than this frame.
        self.bind_all("<Motion>", self._on_mouse_motion, add="+")

        if not self.embedded:
            # Standalone, this widget owns the keyboard. Embedded, the host
            # routes keys in instead -- binding here as well would record
            # every keystroke twice.
            toplevel = self.winfo_toplevel()
            toplevel.bind("<KeyPress>", self.handle_keypress)
            toplevel.bind("<KeyRelease>", self.handle_keyrelease)

        self._refresh_mode_styles()

    def _build_results(self):
        """The panel that replaces the typing area when a run ends."""
        self.results_frame = tk.Frame(self, bg=BG_PANEL)
        body = tk.Frame(self.results_frame, bg=BG_PANEL)
        body.pack(fill="both", expand=True, padx=30,
                  pady=(16, 6) if self.embedded else (26, 6))

        # Embedded, the panel shares its view with the session list and the
        # action row, so the same three stats are set a size down -- at the
        # standalone size the column outgrows the panel and takes the buttons
        # under it off the bottom edge.
        big, small = ((36, 18) if self.embedded else (44, 22))
        stats = tk.Frame(body, bg=BG_PANEL)
        stats.pack(side="left", anchor="center", padx=(0, 28))
        self.wpm_label = self._stat_block(stats, "wpm", ui(big, bold=True),
                                          ACCENT)
        self.acc_label = self._stat_block(stats, "accuracy",
                                          ui(small, bold=True), FG_TEXT)
        self.chars_label = self._stat_block(stats, "characters",
                                            ui(small, bold=True), FG_TEXT)

        self.chart = WpmChart(body, height=170 if self.embedded else 250)
        self.chart.pack(side="left", fill="both", expand=True)

        btn_row = tk.Frame(self.results_frame, bg=BG_PANEL)
        btn_row.pack(pady=(0, 12 if self.embedded else 20))

        restart_btn = tk.Label(btn_row, text="  new test (tab)  ", fg=BG,
                               bg=ACCENT, font=ui(12, bold=True),
                               cursor="hand2", padx=6, pady=6)
        restart_btn.pack(side="left", padx=6)
        restart_btn.bind("<Button-1>", lambda e: self._new_test())

        if self.embedded:
            # the host takes the samples itself, so there is nothing to export
            # here; this line reports what it did with them
            self.result_note = tk.Label(btn_row, text="", fg=FG_LABEL,
                                        bg=BG_PANEL, font=ui(12), padx=10)
            self.result_note.pack(side="left", padx=6)
        else:
            self.result_note = None
            export_btn = tk.Label(btn_row, text="  export samples...  ",
                                  fg=FG_TEXT, bg=BG_PANEL, font=ui(12),
                                  cursor="hand2", padx=6, pady=6,
                                  highlightthickness=1,
                                  highlightbackground=ACCENT_DIM)
            export_btn.pack(side="left", padx=6)
            export_btn.bind("<Button-1>", lambda e: self._export_samples())

    def _stat_block(self, parent, caption, font, colour):
        """One caption-over-value stat in the results column."""
        tk.Label(parent, text=caption, fg=FG_LABEL, bg=BG_PANEL,
                 font=ui(12)).pack(anchor="w")
        value = tk.Label(parent, text="", fg=colour, bg=BG_PANEL, font=font)
        value.pack(anchor="w", pady=(0, 8 if self.embedded else 14))
        return value

    def set_result_note(self, text):
        """Say what the host did with the samples, on the results screen."""
        if self.result_note is not None:
            self.result_note.configure(text=text)

    # ------------------------------------------------------------------
    # Host-facing controls
    # ------------------------------------------------------------------
    def reset(self):
        """Abandon whatever is on screen and deal a fresh test."""
        self._new_test()

    def appear(self, delay=0, duration=320):
        """Fade the words up, for a host that animates its view changes.

        The text colour comes from a Text tag rather than the widget, so --
        like the phrase capture widget -- this fades the tag, not the
        background.
        """
        bg = self.cget("bg")

        def frame(p):
            self.text.tag_configure("untyped", foreground=blend(bg, FG_DIM, p))

        self.anim.tween(duration, frame, delay=delay, key="appear")

    def _set_typing_focus_mode(self, active):
        """Strip the chrome back to just the words while a run is under way."""
        toplevel = self.winfo_toplevel()
        if active:
            self.controls_frame.place_forget()
            if self.bottom_frame is not None:
                self.bottom_frame.place_forget()
            self.text.configure(cursor="none")
            toplevel.configure(cursor="none")
        else:
            self.controls_frame.place(relx=0.5, rely=self.controls_rely,
                                      anchor="n")
            if self.bottom_frame is not None:
                self.bottom_frame.place(**self.bottom_place)
            self.text.configure(cursor="arrow")
            toplevel.configure(cursor="")

    def _on_mouse_motion(self, _event):
        # bind_all fires wherever the pointer is, including while this widget
        # sits on a tab nobody is looking at
        if self.winfo_ismapped():
            self._set_typing_focus_mode(False)

    # ------------------------------------------------------------------
    # Toggle / mode / option handlers
    # ------------------------------------------------------------------
    def _toggle_punctuation(self):
        self.punctuation = not self.punctuation
        self.punct_label.config(fg=ACCENT if self.punctuation else FG_DIM)
        self._new_test()

    def _toggle_numbers(self):
        self.numbers = not self.numbers
        self.numbers_label.config(fg=ACCENT if self.numbers else FG_DIM)
        self._new_test()

    def _on_mode_click(self, name):
        if name == "custom":
            # custom mode always opens the editor, so the text can be changed
            self._open_custom_dialog()
            return
        if self.mode == name:
            return
        self.mode = name
        self._refresh_mode_styles()
        self._rebuild_options()
        self._new_test()

    def _refresh_mode_styles(self):
        for name, lbl in self.mode_labels.items():
            active = name == self.mode
            lbl.config(fg=ACCENT if active else FG_DIM,
                       font=ui(12, bold=True) if active else ui(12))

    def _rebuild_options(self):
        for w in self.options_frame.winfo_children():
            w.destroy()

        def option(text, chosen, command):
            lbl = tk.Label(self.options_frame, text=text,
                           fg=ACCENT if chosen else FG_LABEL, bg=BG_PANEL,
                           font=ui(12), cursor="hand2", padx=6)
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e: command())

        if self.mode == "time":
            for v in TEST_DURATIONS:
                option(str(v), v == self.test_duration,
                       lambda vv=v: self._set_duration(vv))
        elif self.mode == "words":
            for v in WORD_COUNTS:
                option(str(v), v == self.word_count,
                       lambda vv=v: self._set_word_count(vv))
        elif self.mode == "quote":
            for v in QUOTE_LENGTHS:
                option(v, v == self.quote_length,
                       lambda vv=v: self._set_quote_length(vv))
        elif self.mode == "custom":
            option("edit text", False, self._open_custom_dialog)
        elif self.mode == "zen":
            tk.Label(self.options_frame, text="∞", fg=FG_LABEL,
                     bg=BG_PANEL, font=ui(12), padx=6).pack(side="left")

    def _set_duration(self, v):
        self.test_duration = v
        self._rebuild_options()
        self._new_test()

    def _set_word_count(self, v):
        self.word_count = v
        self._rebuild_options()
        self._new_test()

    def _set_quote_length(self, v):
        self.quote_length = v
        self._rebuild_options()
        self._new_test()

    def _open_custom_dialog(self):
        toplevel = self.winfo_toplevel()
        dialog = tk.Toplevel(toplevel)
        dialog.title("Custom text")
        dialog.configure(bg=BG_PANEL)
        dialog.geometry("560x320")
        dialog.transient(toplevel)

        tk.Label(dialog, text="Enter the text you want to type:", fg=FG_TEXT,
                 bg=BG_PANEL, font=ui(12, bold=True)).pack(
                     anchor="w", padx=16, pady=(16, 6))

        text_box = tk.Text(dialog, wrap="word", bg=BG, fg=FG_TEXT,
                           insertbackground=ACCENT, font=mono(TYPE_FONT_SIZE),
                           height=10, borderwidth=0, highlightthickness=1,
                           highlightbackground=ACCENT_DIM, padx=10, pady=10)
        text_box.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        text_box.insert("1.0", self.custom_text or DEFAULT_CUSTOM_TEXT)
        text_box.focus_set()

        btn_row = tk.Frame(dialog, bg=BG_PANEL)
        btn_row.pack(pady=(0, 16))

        def use_text():
            content = text_box.get("1.0", "end").strip()
            self.custom_text = content if content else DEFAULT_CUSTOM_TEXT
            self.mode = "custom"
            self._refresh_mode_styles()
            self._rebuild_options()
            dialog.destroy()
            self._new_test()

        use_btn = tk.Label(btn_row, text="  use this text  ", fg=BG, bg=ACCENT,
                           font=ui(12, bold=True), cursor="hand2", padx=6,
                           pady=6)
        use_btn.pack(side="left", padx=6)
        use_btn.bind("<Button-1>", lambda e: use_text())

        cancel_btn = tk.Label(btn_row, text="  cancel  ", fg=FG_TEXT,
                              bg=BG_PANEL, font=ui(12), cursor="hand2",
                              padx=6, pady=6, highlightthickness=1,
                              highlightbackground=ACCENT_DIM)
        cancel_btn.pack(side="left", padx=6)
        cancel_btn.bind("<Button-1>", lambda e: dialog.destroy())

    # ------------------------------------------------------------------
    # Word-list generation (punctuation / numbers modifiers)
    # ------------------------------------------------------------------
    def _build_word_list(self, n):
        words = [random.choice(WORDLIST) for _ in range(n)]

        if self.numbers:
            for i in range(len(words)):
                if random.random() < 0.12:
                    words[i] = str(random.randint(1, 9999))

        if self.punctuation:
            # sprinkle commas / semicolons mid-list
            for i in range(len(words) - 1):
                if random.random() < 0.10:
                    words[i] = words[i] + random.choice([",", ";"])
            # break into sentence-like chunks: capitalize the first word,
            # end each chunk with terminal punctuation
            idx = 0
            while idx < len(words):
                chunk_len = random.randint(6, 12)
                start = idx
                end = min(idx + chunk_len, len(words))
                first = words[start]
                if first:
                    words[start] = first[0].upper() + first[1:]
                idx = end
                if end - 1 < len(words):
                    last = words[end - 1]
                    if last and last[-1] in ",;":
                        last = last[:-1]
                    words[end - 1] = last + random.choice([".", ".", ".",
                                                           "!", "?"])
            if words and words[-1][-1] not in ".!?":
                words[-1] = words[-1] + "."

        return words

    # ------------------------------------------------------------------
    # Test lifecycle
    # ------------------------------------------------------------------
    def _cancel_jobs(self):
        for attr in ("timer_job", "caret_job", "caret_blink_job",
                     "caret_retry_job"):
            job = getattr(self, attr)
            if job is not None:
                self.after_cancel(job)
                setattr(self, attr, None)

    def _new_test(self):
        self._cancel_jobs()
        self._set_typing_focus_mode(False)

        self.state = "idle"
        self.pos = 0
        self.typed_correct = 0
        self.typed_incorrect = 0
        self.start_perf = None
        self.raw_events = []
        self.session_perf0 = None
        self.held_keys = set()
        self.display_scroll_line = 0
        self.wpm_history = []
        self._last_sample_t = 0.0
        self._last_sample_typed = 0
        self._last_sample_incorrect = 0
        self.caret_position = None
        self.caret_visible = True
        self.caret_blink_level = 0
        self.caret_blink_direction = 1

        if self.mode == "time":
            self.words = self._build_word_list(300)
            self.full_text = " ".join(self.words)
        elif self.mode == "words":
            self.words = self._build_word_list(self.word_count)
            self.full_text = " ".join(self.words)
        elif self.mode == "quote":
            quote = random.choice(QUOTES[self.quote_length])
            self.full_text = quote
            self.words = quote.split()
        elif self.mode == "custom":
            text = (self.custom_text or DEFAULT_CUSTOM_TEXT).strip()
            self.full_text = text
            self.words = text.split()
        elif self.mode == "zen":
            self.words = self._build_word_list(1000)
            self.full_text = " ".join(self.words)

        self.results_frame.place_forget()
        self.set_result_note("")
        self.chart.clear()
        self.status_row.place(relx=.015, rely=self.status_rely, anchor="sw")
        self.text.place(relx=0.5, rely=self.text_rely, anchor="center",
                        relwidth=1.0)

        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.full_text)
        self.text.tag_add("untyped", "1.0", "end")
        self.text.config(state="disabled")
        self.text.yview_moveto(0)
        self.text.update_idletasks()

        self.timer_label.config(text=self._idle_timer_text())
        self._highlight_cursor()

    def _idle_timer_text(self):
        if self.mode == "time":
            return str(self.test_duration)
        if self.mode == "words":
            return f"0/{self.word_count}"
        return "0s"

    def _start_if_needed(self):
        if self.state == "idle":
            self.state = "running"
            self.start_perf = time.perf_counter()
            self.session_perf0 = self.start_perf
            self._tick_timer()

    def _tick_timer(self):
        if self.state != "running":
            return
        elapsed = time.perf_counter() - self.start_perf
        self._flush_samples(elapsed)

        if self.mode == "time":
            remaining = max(0, self.test_duration - elapsed)
            self.timer_label.config(
                text=str(int(remaining) + (1 if remaining % 1 else 0)))
            if remaining <= 0:
                self._finish_test()
                return
        elif self.mode == "words":
            total = self.word_count
            if self.pos >= len(self.full_text):
                completed = total
            else:
                completed = self.full_text[:self.pos].count(" ")
            self.timer_label.config(text=f"{min(completed, total)}/{total}")
        else:
            self.timer_label.config(text=f"{int(elapsed)}s")

        self.timer_job = self.after(100, self._tick_timer)

    def _flush_samples(self, elapsed, final=False):
        """Close out every whole second that has passed since the last sample.

        Sampling on second boundaries is what makes the raw series meaningful:
        each point is the speed of that one second, so it stays jagged, while
        the net series is cumulative and settles as the run goes on. A final
        partial second is only kept if it is long enough not to read as a
        spike.
        """
        while elapsed - self._last_sample_t >= 1.0:
            self._record_sample(self._last_sample_t + 1.0)
        if final and elapsed - self._last_sample_t >= 0.35:
            self._record_sample(elapsed)

    def _record_sample(self, t):
        window = t - self._last_sample_t
        if window <= 0:
            return
        typed = self.typed_correct + self.typed_incorrect
        self.wpm_history.append({
            "t": t,
            "wpm": (self.typed_correct / 5.0) / (t / 60.0),
            "raw": ((typed - self._last_sample_typed) / 5.0) / (window / 60.0),
            "errors": self.typed_incorrect - self._last_sample_incorrect,
        })
        self._last_sample_t = t
        self._last_sample_typed = typed
        self._last_sample_incorrect = self.typed_incorrect

    def _finish_test(self, user_stopped=False):
        if self.state != "running":
            return
        self.state = "finished"
        self._set_typing_focus_mode(False)
        self._cancel_jobs()

        elapsed = time.perf_counter() - (self.start_perf or time.perf_counter())
        self._flush_samples(elapsed, final=True)

        elapsed_minutes = max(1e-6, elapsed / 60.0)
        total_typed = self.typed_correct + self.typed_incorrect
        wpm = (self.typed_correct / 5.0) / elapsed_minutes
        accuracy = ((self.typed_correct / total_typed * 100.0)
                    if total_typed else 0.0)

        self.text.place_forget()
        # the countdown would otherwise show through the widened results panel
        self.status_row.place_forget()
        self.wpm_label.config(text=f"{wpm:.0f}")
        self.acc_label.config(text=f"{accuracy:.1f}%")
        self.chars_label.config(text=str(total_typed))
        self.chart.show(self.wpm_history)
        # Embedded, the panel is placed below the mode controls rather than
        # over the middle of the widget: it is nearly as tall as the space it
        # has, and centring it would put its top edge through the pills.
        if self.embedded:
            self.results_frame.place(relx=0.5, rely=0.57, anchor="center",
                                     relwidth=0.96, relheight=0.86)
        else:
            self.results_frame.place(relx=0.5, rely=0.5, anchor="center",
                                     relwidth=0.92, relheight=0.78)

        self._emit_samples({
            "wpm": wpm,
            "accuracy": accuracy,
            "characters": total_typed,
            "seconds": elapsed,
            "stopped_early": user_stopped,
        })

    def samples(self):
        """This run, cut into pipeline-schema enrollment samples."""
        return split_into_samples(self.raw_events)

    def _emit_samples(self, summary):
        if self.on_finish is None:
            return
        self.on_finish(self.samples(), summary)

    # ------------------------------------------------------------------
    # Caret
    # ------------------------------------------------------------------
    def _highlight_cursor(self):
        self.caret_retry_job = None
        start = f"1.0+{self.pos}c"
        display_count = self.text.count("1.0", start, "displaylines")
        if isinstance(display_count, tuple):
            display_count = display_count[0]
        display_line = (display_count or 0) + 1
        if display_line > self.display_scroll_line + 2:
            self.text.yview_scroll(
                display_line - self.display_scroll_line - 2, "units")
            self.display_scroll_line = display_line - 2

        bbox = self.text.bbox(start)
        if bbox is None:
            self.text.see(start)
            bbox = self.text.bbox(start)
        if bbox is None:
            # The text has no geometry yet -- it is on a tab that has never
            # been shown, or the window has not been mapped. Retry on a timer
            # rather than after_idle, which would spin the CPU for as long as
            # that stays true.
            self.caret_retry_job = self.after(100, self._highlight_cursor)
            return

        target_x, target_y, _, target_height = bbox
        target_x += CARET_X_OFFSET
        target_y += CARET_Y_OFFSET
        old_position = self.caret_position
        self.caret_position = (target_x, target_y, target_height)
        if old_position is None:
            self.caret.place(x=target_x, y=target_y, width=CARET_WIDTH,
                             height=target_height)
        else:
            self._animate_caret(old_position, self.caret_position)
        self._schedule_caret_blink()

    def _animate_caret(self, old_position, target_position, step=0):
        if self.caret_job is not None:
            self.after_cancel(self.caret_job)
        if step >= CARET_ANIMATION_STEPS:
            x, y, height = target_position
            self.caret.place(x=x, y=y, width=CARET_WIDTH, height=height)
            self.caret_job = None
            return
        progress = (step + 1) / CARET_ANIMATION_STEPS
        progress = progress * progress * (3 - 2 * progress)
        x = old_position[0] + (target_position[0] - old_position[0]) * progress
        y = old_position[1] + (target_position[1] - old_position[1]) * progress
        height = target_position[2]
        self.caret.place(x=int(x), y=int(y), width=CARET_WIDTH, height=height)
        self.caret_job = self.after(
            CARET_ANIMATION_DELAY_MS,
            lambda: self._animate_caret(old_position, target_position, step + 1),
        )

    def _schedule_caret_blink(self):
        if self.state != "idle":
            return
        if self.caret_blink_job is not None:
            self.after_cancel(self.caret_blink_job)
        self.caret_visible = True
        self.caret_blink_level = 0
        self.caret_blink_direction = 1
        self.caret.configure(bg=ACCENT)
        self.caret.place_configure(width=CARET_WIDTH)
        self.caret_blink_job = self.after(530, self._blink_caret)

    def _blink_caret(self):
        self.caret_blink_level += self.caret_blink_direction
        if self.caret_blink_level >= CARET_BLINK_STEPS:
            self.caret_blink_level = CARET_BLINK_STEPS
            self.caret_blink_direction = -1
        elif self.caret_blink_level <= 0:
            self.caret_blink_level = 0
            self.caret_blink_direction = 1

        fade = self.caret_blink_level / CARET_BLINK_STEPS
        self.caret.configure(bg=blend(ACCENT, self.cget("bg"), fade))
        self.caret_blink_job = self.after(CARET_BLINK_DELAY_MS,
                                          self._blink_caret)

    def _stop_caret_blink(self):
        if self.caret_blink_job is not None:
            self.after_cancel(self.caret_blink_job)
            self.caret_blink_job = None
        self.caret.configure(bg=ACCENT)
        self.caret.place_configure(width=CARET_WIDTH)

    # ------------------------------------------------------------------
    # Keystroke handling (drives both the typing test AND raw capture)
    # ------------------------------------------------------------------
    @staticmethod
    def _printable_or_space(keysym, char):
        return (len(char) == 1 and char.isprintable()) or keysym == "space"

    def handle_keypress(self, event):
        if not self.active:
            return None
        keysym, char = event.keysym, event.char

        if keysym == "Tab":
            self._new_test()
            # Tab would otherwise travel on to Tk's focus traversal and move
            # the keyboard somewhere else entirely, mid-test
            return "break"
        if keysym == "Escape":
            self._finish_test(user_stopped=True)
            return None
        if keysym == "Return":
            if self.mode == "zen" and self.state == "running":
                self._finish_test(user_stopped=True)
            return None
        if self.state == "finished":
            return None

        if self.state == "idle":
            self._stop_caret_blink()

        if keysym == "BackSpace":
            if self.pos > 0:
                self.pos -= 1
                idx, idx_next = f"1.0+{self.pos}c", f"1.0+{self.pos + 1}c"
                self.text.config(state="normal")
                self.text.tag_remove("correct", idx, idx_next)
                self.text.tag_remove("incorrect", idx, idx_next)
                self.text.tag_add("untyped", idx, idx_next)
                self.text.config(state="disabled")
                self._highlight_cursor()
            return None

        if not self._printable_or_space(keysym, char):
            return None

        key = " " if keysym == "space" else char
        if key in self.held_keys:
            # auto-repeat from a held key: a second keydown with no keyup
            # between would pair the wrong two timestamps into one dwell
            return None

        self._set_typing_focus_mode(True)
        if self.state == "idle" and self.on_typing_start:
            # first keystroke of the run: let the host finish any in-flight
            # animation before it can delay a later keystroke's timestamp
            self.on_typing_start()
        self._start_if_needed()
        if self.state != "running":
            return None

        self.held_keys.add(key)
        t = time.perf_counter() - self.session_perf0
        self.raw_events.append({"key": key, "type": "down", "t": round(t, 6)})

        if self.pos >= len(self.full_text):
            self._finish_test()
            return None

        expected = self.full_text[self.pos]
        idx, idx_next = f"1.0+{self.pos}c", f"1.0+{self.pos + 1}c"
        self.text.config(state="normal")
        self.text.tag_remove("untyped", idx, idx_next)
        if key == expected or (expected == "\n" and key == " "):
            self.text.tag_add("correct", idx, idx_next)
            self.typed_correct += 1
        else:
            self.text.tag_add("incorrect", idx, idx_next)
            self.typed_incorrect += 1
        self.text.config(state="disabled")

        self.pos += 1
        self._highlight_cursor()

        if self.pos >= len(self.full_text):
            self._finish_test()
        return None

    def handle_keyrelease(self, event):
        if not self.active:
            return None
        keysym, char = event.keysym, event.char
        if keysym == "BackSpace" or not self._printable_or_space(keysym, char):
            return None
        key = " " if keysym == "space" else char
        self.held_keys.discard(key)
        if self.state != "running" or self.session_perf0 is None:
            return None
        t = time.perf_counter() - self.session_perf0
        self.raw_events.append({"key": key, "type": "up", "t": round(t, 6)})
        return None

    # ------------------------------------------------------------------
    # Export (standalone only -- an embedding host takes the samples itself)
    # ------------------------------------------------------------------
    def _export_samples(self):
        samples = self.samples()
        if not samples:
            messagebox.showinfo(
                "Nothing to export",
                "Type a longer test first: a sample needs at least "
                f"{MIN_KEYS} keystrokes.")
            return

        user_id = simpledialog.askstring(
            "User ID", "Enter a user ID for these samples:")
        if not user_id:
            return

        record = {
            "user_id": user_id.strip().replace(" ", "_"),
            "phrase": FREE_TEXT_LABEL,
            "samples": samples,
        }

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=f"{record['user_id']}.json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        messagebox.showinfo(
            "Saved",
            f"{len(samples)} sample(s) saved to:\n{path}\n\n"
            "This file is in the schema the pipeline reads, so it can be "
            "dropped straight into the enrollment folder.")


def main():
    root = tk.Tk()
    root.title("keyloggd - typing test")
    root.configure(bg=BG)
    root.geometry("1200x800")
    root.minsize(900, 600)
    init_fonts(root)
    TypingTest(root).pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
