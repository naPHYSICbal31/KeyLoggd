"""
app.py

The main window for the keystroke-behaviour project: a Tkinter frontend over
keyloggd.pipeline, in a warm charcoal / cream / terracotta dark theme.

Three views:

  enroll    Profile setup, and the one place a person's behaviour is
            recorded. It is the Monkeytype-style typing test (ui.typing_test)
            rather than a phrase typed over and over, because the rhythm
            worth enrolling is the one someone types with, not the one they
            fall into on the twelfth repetition of a memorised line. Every
            keydown/keyup is timestamped with time.perf_counter(); a finished
            run is cut into fixed-width samples (pipeline.samples) and
            written to a per-user JSON file in the schema the rest of the
            pipeline reads:

                {"user_id": ..., "phrase": ..., "samples": [{"events": [...]}]}

  identify  Type a phrase (or load a JSON file) and run it through the real
            pipeline -- signal_construction -> fft_features -> classifier /
            identify_sample -- to answer "whose typing behaviour is this?",
            including the open-set UNRECOGNIZED decision.

  explain   The same pipeline, one phase at a time, with the real intermediate
            numbers charted at each step: raw dwell/flight, zero-padding, the
            FFT magnitude spectrum, the 14-number feature vector, z-scores,
            and the distance-to-template decision (see ui.explain).

Identify and explain still type a fixed phrase: both are answering a question
about one attempt, where holding the text constant is the point.

The heavy numpy work (dataset build, EER threshold sweep) happens on a worker
thread so the UI never blocks; results come back through a queue that the Tk
main loop polls.

Run:
    python main.py                       # with the launch splash
    python main.py --no-splash           # straight to the window
    python main.py --data-dir path/to/enrolled

The splash (ui.splash) is shown by default and is not just a curtain: it spends
its time importing numpy and the pipeline modules and building the enrolled
feature matrix on a worker thread, then hands the result to the app, so the
identify view is ready the moment the window opens. --no-splash skips it, which
is what tests and quick restarts want.

Motion: the UI animates (see ui.anim) on the same thread that records the
signal, which costs some timing fidelity. Measured on this machine, against a
synthetic 40 ms/60 ms cadence: flight times are unaffected (sd 0.3 ms vs
0.25 ms with motion off), dwell times pick up ~3 ms of jitter and up to ~16 ms
worst case, on top of a ~5 ms handler bias that is there either way. That is
well inside human dwell variability, but for the cleanest possible enrollment
data run with motion off:

    KEYLOGGD_NO_ANIM=1 python main.py

Requires: tkinter + numpy (the pipeline modules are imported lazily, on first
use or by the splash).
"""

import argparse
import json
import math
import os
import queue
import random
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

from keyloggd.paths import SYNTHETIC_DIR
from keyloggd.pipeline.samples import MIN_KEYS
from keyloggd.ui.anim import (Animator, blend, collect_fade_targets,
                              ease_out_back, ease_out_cubic, fade_in_subtree,
                              local_progress, reveal_text)
from keyloggd.ui.explain import ExplainView, build_phases
from keyloggd.ui.theme import (ACCENT, BG, BG_INPUT, BG_PANEL, FG_CORRECT,
                               FG_DIM, FG_INCORRECT, FG_LABEL, FG_TEXT, OK,
                               WARN, PillButton, RoundedPanel, Spinner,
                               init_fonts, mono, ui)
from keyloggd.ui.typing_test import FREE_TEXT_LABEL, TypingTest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Candidate phrases. Downstream feature extraction works on raw dwell/flight
# timing and doesn't care which literal phrase was typed, so mixing phrases
# across samples/users is fine -- and is a useful robustness test.
PHRASES = [
    "the quick brown fox",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
    "sphinx of black quartz judge my vow",
    "waltz nymph for quick jigs vex bud",
    "bright vixens jump dozy fowl quack",
    "jinxed wizards pluck ivy from the big quilt",
    "two driven jocks help fax my big quiz",
    "quick zephyrs blow vexing daft jim",
    "my ex pub quiz crowd finally won jackpot",
    "grumpy wizards make toxic brew for the jovial queen",
]

MIN_SAMPLES_SUGGESTED = 12
DEFAULT_DATA_DIR = SYNTHETIC_DIR
TYPE_FONT_SIZE = 21


_PIPELINE = None


def load_pipeline():
    """Import numpy and the project's pipeline modules once, and return their
    handles as a dict.

    Kept at module level (rather than on the app) so the splash screen can do
    this import on its worker thread while it is on screen -- it is the slow
    part of startup -- and hand the result straight to the app.
    """
    global _PIPELINE
    if _PIPELINE is None:
        import numpy as np

        from keyloggd.pipeline.classifier import (ZScoreScaler, build_dataset, build_templates,
                                knn_predict, template_distance)
        from keyloggd.pipeline.fft_features import (FFT_LEN, compute_spectrum,
                                  spectral_features, zero_pad)
        from keyloggd.pipeline.identify_sample import (UNRECOGNIZED, compute_open_set_threshold,
                                     fuse, identify, load_unknown_samples)
        from keyloggd.pipeline.signal_construction import sample_to_signal
        from keyloggd.pipeline.timing_features import (FEATURE_NAMES, feature_vector,
                                    signal_feature_vector)

        _PIPELINE = {
            "np": np,
            "build_dataset": build_dataset,
            "feature_vector": feature_vector,
            "signal_feature_vector": signal_feature_vector,
            "FEATURE_NAMES": FEATURE_NAMES,
            "identify": identify,
            "fuse": fuse,
            "load_unknown_samples": load_unknown_samples,
            "compute_open_set_threshold": compute_open_set_threshold,
            "sample_to_signal": sample_to_signal,
            "UNRECOGNIZED": UNRECOGNIZED,
            # the explainer walks the same steps one at a time, so it needs
            # the intermediate stages the identify path calls internally
            "FFT_LEN": FFT_LEN,
            "zero_pad": zero_pad,
            "compute_spectrum": compute_spectrum,
            "spectral_features": spectral_features,
            "ZScoreScaler": ZScoreScaler,
            "build_templates": build_templates,
            "knn_predict": knn_predict,
            "template_distance": template_distance,
        }
    return _PIPELINE


# ---------------------------------------------------------------------------
# Keystroke capture widget
# ---------------------------------------------------------------------------

class TypingCapture(tk.Frame):
    """Phrase display + per-keystroke timing capture.

    Renders the target phrase with per-character colouring (cream = correct,
    soft red = wrong, dim = not yet reached) and records a down/up pair per
    keystroke, timestamped relative to the first keydown of the attempt.

    Backspace pops the most recent completed keystroke off the record rather
    than wiping the attempt: a corrected
    repetition still contributes real timing data, including the natural
    pause before catching a mistake (itself a meaningful behavioural feature).
    """

    CARET_WIDTH = 2
    CARET_GLIDE_MS = 90        # long enough to read as motion, short enough
                               # to stay ahead of a fast typist
    IDLE_BEFORE_BLINK_MS = 800
    BLINK_PERIOD_MS = 1300
    FLASH_MS = 240
    PROGRESS_MS = 260

    def __init__(self, parent, on_complete, on_progress=None,
                 on_typing_start=None, height=2, panel_bg=BG_PANEL):
        super().__init__(parent, bg=panel_bg)
        self.on_complete = on_complete
        self.on_progress = on_progress
        self.on_typing_start = on_typing_start
        self.active = False
        self.panel_bg = panel_bg

        self.phrase = ""
        self.pos = 0
        self.typed = []          # chars as typed, parallel to self.entries
        self.pending = {}        # key -> keydown t, awaiting its keyup
        self.entries = []        # completed {"key","down","up"}, in order
        self.t0 = None           # perf_counter origin for this attempt

        self.anim = Animator(self)
        # the idle blink runs forever, so it gets its own slower clock rather
        # than holding the main 60fps loop open while nothing else moves
        self.blink_anim = Animator(self, interval=33)
        self.caret_x = None      # current (animated) caret x, None = unplaced
        self.flash_tags = set()
        self.progress = 0.0      # drawn fraction of the progress underline
        self.progress_items = None   # (track, fill) canvas item ids

        self.text = tk.Text(
            self, wrap="word", bg=panel_bg, fg=FG_DIM,
            font=mono(TYPE_FONT_SIZE), height=height, borderwidth=0,
            highlightthickness=0, padx=18, pady=10, cursor="arrow",
            spacing1=4, spacing3=4, insertwidth=0,
        )
        self.text.pack(fill="x")
        self.text.tag_configure("correct", foreground=FG_CORRECT)
        self.text.tag_configure("incorrect", foreground=FG_INCORRECT,
                                underline=True)
        self.text.tag_configure("untyped", foreground=FG_DIM)
        self.text.config(state="disabled")

        self.caret = tk.Frame(self.text, bg=ACCENT, width=self.CARET_WIDTH,
                              bd=0, highlightthickness=0)
        # measured once, not per keystroke: three cget() round-trips inside the
        # key handler is work the timing measurement does not need to carry
        self.caret_inset = self._measure_caret_inset()

        # progress underline: fills as the phrase is completed, so there is a
        # visible sense of "how far in am I" without reading the counter
        self.progress_bar = tk.Canvas(self, height=3, bg=panel_bg, bd=0,
                                      highlightthickness=0)
        self.progress_bar.pack(fill="x", padx=18, pady=(4, 0))
        self.progress_bar.bind("<Configure>", lambda e: self._draw_progress())

    # -- phrase / reset ------------------------------------------------------

    def set_phrase(self, phrase):
        self.phrase = phrase
        self.reset()

    def reset(self):
        self.pos = 0
        self.typed = []
        self.pending = {}
        self.entries = []
        self.t0 = None
        self._clear_flashes()
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.phrase)
        self.text.tag_add("untyped", "1.0", "end")
        self.text.config(state="disabled")
        self.caret_x = None          # a fresh phrase snaps rather than glides
        self._place_caret()
        self._report_progress()
        self._schedule_blink()

    # -- geometry / caret ---------------------------------------------------

    def _measure_caret_inset(self):
        """The gap between the Text widget's outer edge and its inner area.

        place() positions a child relative to the *inner* area (inside
        highlightthickness + borderwidth + padx/pady), while bbox() reports
        character coordinates measured from the widget's *outer* edge. The
        inset therefore has to come off a bbox before it can be used as a
        place() coordinate -- otherwise the padding is counted twice and the
        caret lands a whole character to the right and pady too low.
        """
        text = self.text
        edge = (int(text.cget("borderwidth"))
                + int(text.cget("highlightthickness")))
        return int(text.cget("padx")) + edge, int(text.cget("pady")) + edge

    def _place_caret(self):
        """Move the caret to the current position, gliding if it is already
        on screen and the move stays on the same line."""
        if not self.phrase:
            self.caret.place_forget()
            return
        idx = f"1.0+{min(self.pos, len(self.phrase) - 1)}c"
        # bbox is normally already valid: typing only changes tags, not the
        # layout. Flushing idle tasks unconditionally here would run a full
        # relayout inside the keystroke handler and delay the keyup that
        # closes the dwell measurement, so only flush when bbox is unusable
        # (first paint, a resize, a phrase change).
        box = self.text.bbox(idx)
        if box is None:
            self.text.update_idletasks()
            box = self.text.bbox(idx)
        if box is None:
            self.caret.place_forget()
            return
        x, y, w, h = box
        # past the last character, sit just after it rather than on top of it
        if self.pos >= len(self.phrase):
            x += w

        inset_x, inset_y = self.caret_inset
        # 1px left of the cell edge, so the caret straddles the gap between
        # the previous character and the next one
        target = x - inset_x - 1
        place_y = y - inset_y

        same_line = (self.caret_x is not None
                     and self.caret.place_info().get("y") == str(place_y))
        if not same_line:
            # a snap must win over a glide still in flight -- otherwise the
            # tween started by the last keystroke keeps running after a reset
            # and drags the caret back to the end of the finished phrase
            self.anim.cancel("caret")
            self.caret_x = target
            self._draw_caret(target, place_y, h)
            return

        start = self.caret_x
        self.caret_x = target

        def frame(p):
            self._draw_caret(start + (target - start) * p, place_y, h)

        self.anim.tween(self.CARET_GLIDE_MS, frame, easing=ease_out_cubic,
                        key="caret")

    def _draw_caret(self, x, y, height):
        self.caret.place(x=round(x), y=y, width=self.CARET_WIDTH, height=height)
        self.caret.lift()

    def _schedule_blink(self):
        """Blink the caret once typing pauses -- the classic 'waiting for you'
        cue. Cancelled on the next keystroke so it never blinks mid-word."""
        def frame(p):
            # cosine so the caret rests at full accent and dips softly out
            amount = 0.5 + 0.5 * math.cos(2 * math.pi * p)
            self.caret.configure(bg=blend(self.panel_bg, ACCENT, amount))

        self.blink_anim.repeat(self.BLINK_PERIOD_MS, frame, key="blink",
                               delay=self.IDLE_BEFORE_BLINK_MS)

    def _stop_blink(self):
        self.blink_anim.cancel("blink")
        self.caret.configure(bg=ACCENT)

    # -- character flash / progress ----------------------------------------

    def _flash_char(self, pos, flash_colour, settle_colour):
        """Briefly brighten the character just typed, then settle it into its
        final colour -- immediate feedback that the keystroke registered."""
        tag = f"flash{pos}"
        idx, idx_next = f"1.0+{pos}c", f"1.0+{pos + 1}c"
        self.text.tag_configure(tag, foreground=flash_colour)
        self.text.tag_add(tag, idx, idx_next)
        self.text.tag_raise(tag)
        self.flash_tags.add(tag)

        def frame(p):
            self.text.tag_configure(
                tag, foreground=blend(flash_colour, settle_colour, p))

        def done():
            self._drop_flash(tag)

        self.anim.tween(self.FLASH_MS, frame, on_done=done, key=tag)

    def _drop_flash(self, tag):
        self.anim.cancel(tag)
        if tag in self.flash_tags:
            self.flash_tags.discard(tag)
            try:
                self.text.tag_delete(tag)
            except tk.TclError:
                pass

    def _clear_flashes(self):
        for tag in list(self.flash_tags):
            self._drop_flash(tag)

    def appear(self, *, delay=0, duration=320):
        """Fade the phrase up out of the panel surface, for a view switch or
        first paint. The phrase colour lives on a Text tag, which a generic
        widget fade cannot reach."""
        def frame(p):
            self.text.tag_configure(
                "untyped", foreground=blend(self.panel_bg, FG_DIM, p))

        self.anim.tween(duration, frame, delay=delay, key="appear")

    def flash_reject(self):
        """Wash the phrase red and fade back to dim: a rejected repetition
        should be unmistakable without a modal dialog."""
        def frame(p):
            self.text.tag_configure(
                "untyped", foreground=blend(FG_INCORRECT, FG_DIM, p))

        self.anim.tween(620, frame, easing=ease_out_cubic, key="reject")

    def _draw_progress(self):
        """Resize the two existing rectangles rather than rebuilding them.

        This redraw runs every animation frame *while the user is typing*, so
        it is the one piece of animation work that could delay the handler
        that timestamps a keystroke. coords() on two items is roughly free;
        delete-and-recreate was not.
        """
        canvas = self.progress_bar
        w = max(canvas.winfo_width(), 1)
        h = max(canvas.winfo_height(), 1)

        if self.progress_items is None:
            track = canvas.create_rectangle(
                0, h - 2, w, h, fill=blend(self.panel_bg, BG, 0.7), outline="")
            fill = canvas.create_rectangle(0, h - 3, 0, h, fill=ACCENT,
                                           outline="", state="hidden")
            self.progress_items = (track, fill)

        track, fill = self.progress_items
        canvas.coords(track, 0, h - 2, w, h)
        if self.progress > 0:
            canvas.coords(fill, 0, h - 3, max(2, w * self.progress), h)
            canvas.itemconfigure(fill, state="normal")
        else:
            canvas.itemconfigure(fill, state="hidden")

    def _animate_progress(self, target):
        start = self.progress

        def frame(p):
            self.progress = start + (target - start) * p
            self._draw_progress()

        self.anim.tween(self.PROGRESS_MS, frame, easing=ease_out_cubic,
                        key="progress")

    # -- key handling -------------------------------------------------------

    @staticmethod
    def _printable_or_space(keysym, char):
        return (len(char) == 1 and char.isprintable()) or keysym == "space"

    def _now(self):
        """Seconds since this attempt's first keystroke, via perf_counter.

        Tk hands us `event.time` as well, which looks tempting here: it is
        stamped by the OS when the key was pressed, so unlike a perf_counter
        reading taken in the handler it cannot be pushed late by UI work on
        the same thread. Measured on Windows, though, event.time comes from
        the GetTickCount clock and only advances in ~15.6 ms steps -- against
        dwell times around 85 ms that quantisation is a bigger error than the
        dispatch delay it would avoid, so perf_counter stays the clock and
        the animations are kept cheap instead (see
        CaptureToolApp._settle_chrome_animations).
        """
        if self.t0 is None:
            self.t0 = time.perf_counter()
            return 0.0
        return time.perf_counter() - self.t0

    def handle_keypress(self, event):
        if not self.active or not self.phrase:
            return
        keysym, char = event.keysym, event.char

        if keysym == "BackSpace":
            self._backspace()
            return
        if keysym == "Escape":
            self.reset()
            return
        if keysym == "Return":
            self._submit()
            return
        if not self._printable_or_space(keysym, char):
            return
        if self.pos >= len(self.phrase):
            return

        key = " " if keysym == "space" else char
        # ignore auto-repeat from a held-down key: a second keydown with no
        # intervening keyup would otherwise inject a garbage timing entry
        if key in self.pending:
            return
        if not self.typed and self.on_typing_start:
            # first keystroke of the attempt: let the app settle any in-flight
            # chrome animation before it can delay a later keystroke
            self.on_typing_start()
        self.pending[key] = self._now()
        self._stop_blink()

        expected = self.phrase[self.pos]
        correct = key == expected
        idx, idx_next = f"1.0+{self.pos}c", f"1.0+{self.pos + 1}c"
        self.text.config(state="normal")
        self.text.tag_remove("untyped", idx, idx_next)
        self.text.tag_add("correct" if correct else "incorrect", idx, idx_next)
        self.text.config(state="disabled")

        if correct:
            self._flash_char(self.pos, ACCENT, FG_CORRECT)
        else:
            self._flash_char(self.pos, blend(FG_INCORRECT, "#FFFFFF", 0.45),
                             FG_INCORRECT)

        self.typed.append(key)
        self.pos += 1
        self._place_caret()
        self._report_progress()
        self._schedule_blink()

    def handle_keyrelease(self, event):
        if not self.active:
            return
        keysym, char = event.keysym, event.char
        if not self._printable_or_space(keysym, char):
            return
        key = " " if keysym == "space" else char
        if key not in self.pending:
            return  # no matching keydown (ignored repeat, or held across a reset)
        down_t = self.pending.pop(key)
        self.entries.append({"key": key, "down": down_t, "up": self._now()})

        # auto-submit once the last keystroke of the phrase has been released,
        # so the final dwell time is part of the recorded sample
        if self.pos >= len(self.phrase) and not self.pending:
            self._submit()

    def _backspace(self):
        if self.pos == 0:
            return
        self.pos -= 1
        if self.typed:
            self.typed.pop()
        if self.entries:
            self.entries.pop()
        self._drop_flash(f"flash{self.pos}")  # no stale flash on an undone char
        idx, idx_next = f"1.0+{self.pos}c", f"1.0+{self.pos + 1}c"
        self.text.config(state="normal")
        self.text.tag_remove("correct", idx, idx_next)
        self.text.tag_remove("incorrect", idx, idx_next)
        self.text.tag_add("untyped", idx, idx_next)
        self.text.config(state="disabled")
        self._stop_blink()
        self._place_caret()
        self._report_progress()
        self._schedule_blink()

    # -- output -------------------------------------------------------------

    def flat_events(self):
        """The {"key","type","t"} event stream signal_construction.py expects,
        in chronological order, one down immediately followed by its up."""
        out = []
        for e in self.entries:
            out.append({"key": e["key"], "type": "down", "t": round(e["down"], 6)})
            out.append({"key": e["key"], "type": "up", "t": round(e["up"], 6)})
        return out

    def _report_progress(self):
        total = len(self.phrase)
        self._animate_progress(self.pos / total if total else 0.0)
        if self.on_progress:
            self.on_progress(self.pos, total)

    def _submit(self):
        typed_text = "".join(self.typed)
        if not typed_text:
            return

        reason = None
        if typed_text != self.phrase:
            reason = "mismatch"
        elif len(self.entries) != len(self.phrase):
            # safety net: a key held across a reset, or an untracked edit,
            # would leave the record misaligned with the visible text
            reason = "untracked"

        sample = None
        if reason is None:
            sample = {
                "events": self.flat_events(),
                "phrase": self.phrase,
                "captured_at": datetime.now().isoformat(timespec="seconds"),
            }

        # clear first, then report: reset() fires the progress callback, which
        # would otherwise overwrite the status line on_complete just set
        self.reset()
        self.on_complete(sample, reason or "ok")


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------

def sample_stats(sample):
    """Dwell/flight summary for one captured sample, in milliseconds.

    Computed here from the raw event list rather than through numpy so the
    enroll view stays dependency-free and instant.
    """
    downs, ups = [], []
    for ev in sample["events"]:
        (downs if ev["type"] == "down" else ups).append(ev["t"])
    n = min(len(downs), len(ups))
    dwell = [(ups[i] - downs[i]) * 1000.0 for i in range(n)]
    flight = [(downs[i + 1] - ups[i]) * 1000.0 for i in range(n - 1)]
    total = ups[-1] if ups else 0.0
    chars = n
    wpm = (chars / 5.0) / (total / 60.0) if total > 0 else 0.0
    return {
        "dwell": dwell,
        "flight": flight,
        "mean_dwell": sum(dwell) / len(dwell) if dwell else 0.0,
        "mean_flight": sum(flight) / len(flight) if flight else 0.0,
        "seconds": total,
        "chars": chars,
        "wpm": wpm,
    }


class SignalStrip(tk.Canvas):
    """Bar plot of the last sample's dwell and flight signals.

    This is the actual discrete signal signal_construction.py builds, drawn
    directly so it is obvious what the pipeline is looking at.
    """

    REVEAL_MS = 520

    def __init__(self, parent, height=104):
        super().__init__(parent, bg=BG_PANEL, height=height, bd=0,
                         highlightthickness=0)
        self.stats = None
        self.reveal = 1.0        # 0 -> bars flat, 1 -> bars at full height
        self.anim = Animator(self)
        self.bind("<Configure>", lambda e: self._draw())

    def show(self, stats):
        """Draw a new sample, growing the bars up out of the baseline in a
        left-to-right sweep so the shape of the rhythm registers."""
        self.stats = stats

        def frame(p):
            self.reveal = p
            self._draw()

        self.anim.tween(self.REVEAL_MS, frame, easing=ease_out_cubic,
                        key="reveal")

    def clear(self):
        self.anim.cancel("reveal")
        self.stats = None
        self.reveal = 1.0
        self._draw()

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        if not self.stats or not self.stats["dwell"]:
            self.create_text(w // 2, h // 2, text="no sample captured yet",
                             fill=FG_DIM, font=ui(10))
            return

        pad = 12
        row_h = (h - pad * 3) // 2
        for row, (key, colour, label) in enumerate(
            [("dwell", ACCENT, "dwell"), ("flight", FG_LABEL, "flight")]
        ):
            series = self.stats[key]
            if not series:
                continue
            top = pad + row * (row_h + pad)
            base = top + row_h
            peak = max(max(series), 1.0)
            # reserve a gutter on each side so the row label and the peak
            # readout never sit on top of the bars
            label_w, gutter_w = 46, 88
            span = w - pad * 2 - label_w - gutter_w
            bar_w = max(2, min(11, span // max(len(series), 1) - 2))
            step = span / max(len(series), 1)
            self.create_text(pad, base, text=label, anchor="sw",
                             fill=blend(BG_PANEL, FG_DIM, self.reveal),
                             font=ui(9))
            for i, v in enumerate(series):
                grow = local_progress(self.reveal, i, len(series), spread=0.6)
                bar_h = max(1, (abs(v) / peak) * row_h * grow)
                x = pad + label_w + i * step
                self.create_rectangle(x, base - bar_h, x + bar_w, base,
                                      fill=blend(BG_PANEL, colour,
                                                 0.25 + 0.75 * grow),
                                      outline="")
            self.create_text(w - pad, base - row_h // 2,
                             text=f"peak\n{peak:.0f} ms", anchor="e",
                             justify="right",
                             fill=blend(BG_PANEL, FG_DIM, self.reveal),
                             font=ui(9))


class DistanceBars(tk.Canvas):
    """Ranked template distances for an identification attempt.

    Shorter bar = closer to that enrolled user's template = more likely them.
    The accept/reject threshold is drawn as a dashed line so an UNRECOGNIZED
    verdict is visually self-explanatory.
    """

    REVEAL_MS = 720

    def __init__(self, parent, height=190):
        super().__init__(parent, bg=BG_PANEL, height=height, bd=0,
                         highlightthickness=0)
        self.ranked = None
        self.threshold = None
        self.winner = None
        self.reveal = 1.0
        self.anim = Animator(self)
        self.bind("<Configure>", lambda e: self._draw())

    def show(self, ranked, threshold, winner):
        """Sweep the bars out from the labels, closest match first, then fade
        the threshold line in over them once the ranking has landed."""
        self.ranked, self.threshold, self.winner = ranked, threshold, winner

        def frame(p):
            self.reveal = p
            self._draw()

        self.anim.tween(self.REVEAL_MS, frame, easing=ease_out_cubic,
                        key="reveal")

    def clear(self):
        self.anim.cancel("reveal")
        self.ranked = None
        self.reveal = 1.0
        self._draw()

    def _draw(self):
        self.delete("all")
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        if not self.ranked:
            self.create_text(w // 2, h // 2,
                             text="distance to each enrolled user appears here",
                             fill=FG_DIM, font=ui(10))
            return

        pad = 14
        label_w = 108
        rows = self.ranked[:8]
        row_h = min(26, (h - pad * 2) // max(len(rows), 1))
        span = w - pad * 2 - label_w - 58
        scale_max = max([d for _, d in rows] + [self.threshold or 0]) * 1.08 or 1.0

        for i, (user, dist) in enumerate(rows):
            y = pad + i * row_h + row_h // 2
            is_winner = user == self.winner
            grow = local_progress(self.reveal, i, len(rows), spread=0.5)
            # the winning bar overshoots a hair before settling, which draws
            # the eye to the row that decided the verdict
            eased = ease_out_back(grow, 0.9) if is_winner else grow
            self.create_text(pad + label_w, y, text=user, anchor="e",
                             fill=blend(BG_PANEL,
                                        FG_TEXT if is_winner else FG_LABEL,
                                        grow),
                             font=ui(11, bold=is_winner))
            # clamped: the winner's overshoot must not run past the panel
            bar_len = max(2, min(span,
                                 (dist / scale_max) * span * max(eased, 0.0)))
            x0 = pad + label_w + 12
            self.create_rectangle(x0, y - 6, x0 + bar_len, y + 6,
                                  fill=ACCENT if is_winner else "#4A4A46",
                                  outline="")
            self.create_text(x0 + bar_len + 8, y, text=f"{dist:.2f}",
                             anchor="w", fill=blend(BG_PANEL, FG_DIM, grow),
                             font=mono(9))

        if self.threshold:
            # hold the threshold back until the ranking has mostly arrived
            fade = max(0.0, (self.reveal - 0.5) / 0.5)
            x = pad + label_w + 12 + (self.threshold / scale_max) * span
            self.create_line(x, pad - 4, x, pad + len(rows) * row_h,
                             fill=blend(BG_PANEL, WARN, fade), dash=(3, 3))
            self.create_text(x, pad + len(rows) * row_h + 10,
                             text=f"threshold {self.threshold:.2f}",
                             anchor="n", fill=blend(BG_PANEL, WARN, fade),
                             font=ui(9))


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class CaptureToolApp:
    def __init__(self, root, preloaded=None, data_dir=None):
        self.root = root
        init_fonts(root)

        root.title("keyloggd - keystroke behaviour capture")
        root.configure(bg=BG)
        root.geometry("1120x780")
        root.minsize(980, 700)

        self.view = "enroll"          # enroll | identify | explain
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.phrase_index = 0

        # enroll state
        self.samples = []             # completed repetitions this session

        # identify state
        self.pipeline = None          # lazily imported module handles
        self.enrolled = None          # (X, y, user_ids)
        self.threshold = None
        self.eer = None
        self.loading = False
        self.attempt_stats = None     # dwell/flight summary of the typed attempt
        self.results_queue = queue.Queue()

        # explain state
        self.explain_showing_walkthrough = False

        self.anim = Animator(root)

        self._build_ui()
        self._snapshot_fade_targets()
        self._set_phrase(0)
        self._show_view("enroll", animate=False)
        self._adopt_preloaded(preloaded)
        self._fade_in_window()
        self.root.after(120, self._poll_worker)

    def _adopt_preloaded(self, preloaded):
        """Take whatever the splash screen managed to load on its way in, so
        the identify view is ready without a second pass over the data."""
        if not preloaded:
            return
        self.pipeline = preloaded.get("pipeline") or self.pipeline
        enrolled = preloaded.get("enrolled")
        if enrolled:
            self._apply_enrolled(enrolled)

    def _fade_in_window(self):
        """Fade the whole window up on launch instead of flashing into place.

        Window alpha is platform-dependent, so a failure here just means the
        window shows immediately -- never that it stays invisible.
        """
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:
            self._appear_view("enroll")
            return

        def frame(p):
            try:
                self.root.attributes("-alpha", p)
            except tk.TclError:
                pass

        self.anim.tween(260, frame, easing=ease_out_cubic, key="window")

        # chrome first, then the view's blocks, so the eye lands on the title
        fade_in_subtree(self.anim, self.header, BG, duration=320, key="header",
                        targets=self.fade_targets.get(id(self.header)))
        fade_in_subtree(self.anim, self.footer, BG, duration=320, delay=260,
                        key="footer",
                        targets=self.fade_targets.get(id(self.footer)))
        self._appear_view("enroll")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=28, pady=(20, 0))
        self.header = header

        tk.Label(header, text="keyloggd", fg=ACCENT, bg=BG,
                 font=ui(20, bold=True)).pack(side="left")
        tk.Label(header, text="   keystroke behaviour capture & identification",
                 fg=FG_DIM, bg=BG, font=ui(11)).pack(side="left", pady=(6, 0))

        # view tabs, as a pill on the right of the header
        tabs = RoundedPanel(header, BG_PANEL, radius=12, padding=5)
        tabs.pack(side="right")
        self.tab_labels = {}
        for name, icon in [("enroll", "◉"), ("identify", "△"),
                           ("explain", "◈")]:
            lbl = tk.Label(tabs.inner, text=f" {icon} {name} ", fg=FG_DIM,
                           bg=BG_PANEL, font=ui(11), cursor="hand2",
                           padx=10, pady=4)
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e, n=name: self._show_view(n))
            self.tab_labels[name] = lbl

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=28, pady=18)
        self.body = body

        self.enroll_view = self._build_enroll_view(body)
        self.identify_view = self._build_identify_view(body)
        self.explain_view = self._build_explain_view(body)

        # footer hints
        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=28, pady=(0, 14))
        self.footer = footer
        self.hint_label = tk.Label(
            footer, fg=FG_DIM, bg=BG, font=ui(10),
            text="just type - a finished test becomes several samples   "
                 "|   tab: new test   |   esc: stop early",
        )
        self.hint_label.pack(side="left")

        # keystrokes are captured at the toplevel and routed to whichever
        # capture widget is visible, so no click-to-focus dance is needed
        self.root.bind("<KeyPress>", self._route_keypress)
        self.root.bind("<KeyRelease>", self._route_keyrelease)

    # -- shared sub-widgets -------------------------------------------------

    def _phrase_bar(self, parent):
        """The phrase picker pill: prev / index / next / random."""
        bar = tk.Frame(parent, bg=BG)
        pill = RoundedPanel(bar, BG_PANEL, radius=12, padding=5)
        pill.pack(side="left")
        inner = pill.inner

        def nav(label, delta):
            lbl = tk.Label(inner, text=label, fg=FG_LABEL, bg=BG_PANEL,
                           font=ui(11), cursor="hand2", padx=9, pady=4)
            lbl.pack(side="left")
            lbl.bind("<Button-1>", lambda e: self._step_phrase(delta))
            return lbl

        nav("◀", -1)
        counter = tk.Label(inner, text="", fg=FG_TEXT, bg=BG_PANEL,
                           font=ui(11), padx=4, pady=4)
        counter.pack(side="left")
        nav("▶", 1)

        shuffle = tk.Label(inner, text="  ⇄ random  ", fg=ACCENT,
                           bg=BG_PANEL, font=ui(11), cursor="hand2",
                           padx=6, pady=4)
        shuffle.pack(side="left")
        shuffle.bind("<Button-1>", lambda e: self._random_phrase())

        status = tk.Label(bar, text="", fg=FG_DIM, bg=BG, font=ui(10))
        status.pack(side="left", padx=14)
        return bar, counter, status

    def _labelled_panel(self, parent, title):
        """A titled panel block; returns the frame children should go into."""
        wrap = tk.Frame(parent, bg=BG)
        tk.Label(wrap, text=title.upper(), fg=FG_DIM, bg=BG,
                 font=ui(9)).pack(anchor="w", pady=(0, 5))
        panel = tk.Frame(wrap, bg=BG_PANEL)
        panel.pack(fill="both", expand=True)
        return wrap, panel

    def _typing_card(self, parent, on_complete, on_progress):
        """The panel that frames a TypingCapture, so all three views present
        the phrase identically."""
        wrap = tk.Frame(parent, bg=BG_PANEL)
        capture = TypingCapture(wrap, on_complete=on_complete,
                                on_progress=on_progress,
                                on_typing_start=self._settle_chrome_animations)
        capture.pack(fill="x", padx=6, pady=10)
        return wrap, capture

    # -- enroll view --------------------------------------------------------

    def _build_enroll_view(self, parent):
        """Profile setup: a typing test, and what it produced.

        Enrollment used to be the same fixed phrase typed over and over. It
        is a typing test now, because what the pipeline needs is the rhythm
        someone types with, and a person repeating one memorised phrase for
        the twelfth time is not typing the way they usually type. A run here
        is longer, the words are unpredictable, and one 30-second test hands
        back several samples at once (see pipeline.samples).
        """
        view = tk.Frame(parent, bg=BG)

        # user id + data dir row
        top = tk.Frame(view, bg=BG)
        top.pack(fill="x")

        tk.Label(top, text="user id", fg=FG_LABEL, bg=BG,
                 font=ui(11)).pack(side="left", padx=(0, 8))
        id_pill = RoundedPanel(top, BG_INPUT, radius=8, padding=4)
        id_pill.pack(side="left")
        self.user_entry = tk.Entry(
            id_pill.inner, bg=BG_INPUT, fg=FG_TEXT, insertbackground=ACCENT,
            font=mono(12), width=20, relief="flat", highlightthickness=0, bd=0,
        )
        self.user_entry.pack(padx=8, pady=4)

        self.enroll_status = tk.Label(top, text="", fg=FG_DIM, bg=BG,
                                      font=ui(10))
        self.enroll_status.pack(side="left", padx=14)

        self.dir_label = tk.Label(top, text="", fg=FG_DIM, bg=BG, font=ui(10))
        self.dir_label.pack(side="right")
        PillButton(top, "change folder", self._choose_data_dir, size=10,
                   bg=BG, padx=13, pady=6).pack(side="right", padx=10)

        # the typing test, as the enrollment input
        type_wrap = tk.Frame(view, bg=BG)
        type_wrap.pack(fill="both", expand=True, pady=(14, 16))
        self.enroll_capture = TypingTest(
            type_wrap, on_finish=self._on_typing_test_finish,
            on_typing_start=self._settle_chrome_animations, embedded=True)
        self.enroll_capture.pack(fill="both", expand=True)

        # lower half: session list | live signal
        lower = tk.Frame(view, bg=BG, height=self.ENROLL_LOWER_HEIGHT)
        lower.pack(fill="x")
        # the typing test above is the thing that should absorb a resize, so
        # this strip keeps the height it was given
        lower.pack_propagate(False)
        lower.columnconfigure(0, weight=3, uniform="cols")
        lower.columnconfigure(1, weight=4, uniform="cols")
        lower.rowconfigure(0, weight=1)

        list_wrap, list_panel = self._labelled_panel(lower, "session samples")
        list_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.samples_text = tk.Text(
            list_panel, bg=BG_PANEL, fg=FG_LABEL, font=mono(10), bd=0,
            highlightthickness=0, padx=12, pady=10, wrap="none", height=6,
            state="disabled", cursor="arrow",
        )
        self.samples_text.pack(fill="both", expand=True)
        self.samples_text.tag_configure("ok", foreground=OK)
        self.samples_text.tag_configure("dim", foreground=FG_DIM)

        sig_wrap, sig_panel = self._labelled_panel(lower, "last sample signal")
        sig_wrap.grid(row=0, column=1, sticky="nsew")
        self.signal_strip = SignalStrip(sig_panel)
        self.signal_strip.pack(fill="both", expand=True, padx=6, pady=6)
        self.sample_stats_label = tk.Label(
            sig_panel, text="", fg=FG_DIM, bg=BG_PANEL, font=mono(10),
            anchor="w", justify="left", padx=14,
        )
        self.sample_stats_label.pack(fill="x", pady=(0, 10))

        # action row
        actions = tk.Frame(view, bg=BG)
        actions.pack(fill="x", pady=(18, 0))
        self.save_btn = PillButton(actions, "save to enrollment folder",
                                   self._save_enrollment, kind="primary")
        self.save_btn.pack(side="left")
        PillButton(actions, "export json...", self._export_json,
                   bg=BG).pack(side="left", padx=10)
        PillButton(actions, "clear session", self._clear_session,
                   bg=BG).pack(side="left")
        self.enroll_count_label = tk.Label(
            actions, text="", fg=FG_DIM, bg=BG, font=ui(11))
        self.enroll_count_label.pack(side="right")

        self._refresh_enroll_counts()
        self._refresh_dir_label()
        # top-to-bottom order drives the staggered appear-in
        self.enroll_blocks = [top, type_wrap, lower, actions]
        return view

    # -- identify view ------------------------------------------------------

    def _build_identify_view(self, parent):
        view = tk.Frame(parent, bg=BG)

        top = tk.Frame(view, bg=BG)
        top.pack(fill="x")
        self.spinner = Spinner(top, bg=BG)
        self.enrolled_label = tk.Label(top, text="enrolled set not loaded",
                                       fg=FG_LABEL, bg=BG, font=ui(11))
        self.enrolled_label.pack(side="left")
        PillButton(top, "reload enrolled set", self._load_enrolled, size=10,
                   bg=BG, padx=13, pady=6).pack(side="right")
        PillButton(top, "identify from file...", self._identify_from_file,
                   size=10, bg=BG, padx=13, pady=6).pack(side="right", padx=10)

        bar, self.identify_counter, self.identify_status = self._phrase_bar(view)
        bar.pack(fill="x", pady=(20, 4))

        type_wrap, self.identify_capture = self._typing_card(
            view, self._on_identify_sample, self._on_identify_progress)
        type_wrap.pack(fill="x", pady=(6, 20))

        lower = tk.Frame(view, bg=BG)
        lower.pack(fill="both", expand=True)
        lower.columnconfigure(0, weight=3, uniform="cols")
        lower.columnconfigure(1, weight=4, uniform="cols")
        lower.rowconfigure(0, weight=1)

        verdict_wrap, verdict_panel = self._labelled_panel(lower, "verdict")
        verdict_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.verdict_label = tk.Label(verdict_panel, text="—", fg=FG_DIM,
                                      bg=BG_PANEL, font=ui(34, bold=True))
        self.verdict_label.pack(pady=(26, 4), padx=16)
        self.verdict_sub = tk.Label(
            verdict_panel, text="type the phrase above to identify a typist",
            fg=FG_DIM, bg=BG_PANEL, font=ui(10), wraplength=280,
            justify="center")
        self.verdict_sub.pack(pady=(0, 10), padx=16)
        self.verdict_detail = tk.Label(
            verdict_panel, text="", fg=FG_LABEL, bg=BG_PANEL, font=mono(10),
            justify="left", anchor="w")
        self.verdict_detail.pack(pady=(0, 18), padx=18, fill="x")

        dist_wrap, dist_panel = self._labelled_panel(
            lower, "distance to enrolled templates")
        dist_wrap.grid(row=0, column=1, sticky="nsew")
        self.distance_bars = DistanceBars(dist_panel)
        self.distance_bars.pack(fill="both", expand=True, padx=6, pady=6)

        self.identify_blocks = [top, bar, type_wrap, lower]
        return view

    # -- explain view -------------------------------------------------------

    def _build_explain_view(self, parent):
        view = tk.Frame(parent, bg=BG)

        top = tk.Frame(view, bg=BG)
        top.pack(fill="x")
        tk.Label(top,
                 text="type the phrase once, then step through exactly what "
                      "the pipeline does with it",
                 fg=FG_LABEL, bg=BG, font=ui(11)).pack(side="left")

        bar, self.explain_counter, self.explain_status = self._phrase_bar(view)
        bar.pack(fill="x", pady=(20, 4))

        type_wrap, self.explain_capture = self._typing_card(
            view, self._on_explain_sample, self._on_explain_progress)
        type_wrap.pack(fill="x", pady=(6, 20))
        self.explain_type_wrap = type_wrap

        self.explain_walkthrough = ExplainView(view,
                                               on_replay=self._explain_retype)

        self.explain_blocks = [top, bar, type_wrap]
        return view

    # ------------------------------------------------------------------
    # View switching / phrase selection
    # ------------------------------------------------------------------

    ENROLL_LOWER_HEIGHT = 168   # px kept for the session list / signal strip
    APPEAR_STAGGER_MS = 65

    def _snapshot_fade_targets(self):
        """Record each block's design colours once, before anything animates.

        Widgets that animate their own colour elsewhere -- status lines, the
        view tabs, the save button, the verdict text -- are left out: fading
        them from a snapshot would let the appear-in overwrite a message or
        state set while it was still running.
        """
        live = (self.enroll_status, self.identify_status, self.explain_status,
                self.enrolled_label, self.save_btn, self.sample_stats_label,
                self.verdict_label, self.verdict_sub, self.verdict_detail,
                *self.tab_labels.values())
        self.fade_targets = {
            id(block): collect_fade_targets(block, skip=live)
            for block in (*self.enroll_blocks, *self.identify_blocks,
                          *self.explain_blocks, self.header, self.footer)
        }

    def _settle_chrome_animations(self):
        """Finish any window-level animation the moment typing starts.

        The appear-in fade touches every widget in a block each frame, which
        is the one effect heavy enough to sit between a keypress and the
        handler that timestamps it. Snapping it to its end state keeps the
        recorded signal clean; the effect has already served its purpose by
        the time anyone starts typing.
        """
        self.anim.cancel_matching(("appear-", "header", "footer", "window"),
                                  finish=True)

    def _capture_for(self, name):
        return {"enroll": self.enroll_capture,
                "identify": self.identify_capture,
                "explain": self.explain_capture}[name]

    def _appear_view(self, name):
        """Stagger the view's blocks in from the background, top to bottom."""
        blocks = {"enroll": self.enroll_blocks,
                  "identify": self.identify_blocks,
                  "explain": self.explain_blocks}[name]
        capture = self._capture_for(name)
        for i, block in enumerate(blocks):
            fade_in_subtree(self.anim, block, BG, duration=300,
                            delay=i * self.APPEAR_STAGGER_MS,
                            key=f"appear-{name}-{i}",
                            targets=self.fade_targets.get(id(block)))
        # the phrase itself is coloured by a Text tag, so it needs its own fade
        capture.appear(delay=2 * self.APPEAR_STAGGER_MS)

    def _show_view(self, name, animate=True):
        self.view = name
        views = {"enroll": self.enroll_view, "identify": self.identify_view,
                 "explain": self.explain_view}
        for other in views.values():
            other.pack_forget()
        views[name].pack(fill="both", expand=True)

        for tab_name, lbl in self.tab_labels.items():
            active = tab_name == name
            start = str(lbl.cget("fg"))
            end = ACCENT if active else FG_DIM
            lbl.configure(font=ui(11, bold=active))
            self.anim.tween(
                180,
                lambda p, l=lbl, s=start, e=end: l.configure(fg=blend(s, e, p)),
                key=f"tab-{tab_name}")

        if animate:
            self._appear_view(name)

        for view_name in views:
            capture = self._capture_for(view_name)
            # the explain view hides its typing card once a walkthrough is up,
            # so it only listens while that card is the thing on screen
            capture.active = (view_name == name
                              and not (view_name == "explain"
                                       and self.explain_showing_walkthrough))
            capture.reset()

        # the typing test carries its own controls, so it gets its own hint
        if name == "enroll":
            self.hint_label.configure(
                text="just type - a finished test becomes several samples   "
                     "|   tab: new test   |   esc: stop early")
        else:
            action = ("identify the typist" if name == "identify"
                      else "walk through the pipeline")
            self.hint_label.configure(
                text=f"type the phrase to {action}   |   backspace: undo a "
                     "keystroke   |   esc: restart the phrase   |   enter: "
                     "submit")

        if (name in ("identify", "explain") and self.enrolled is None
                and not self.loading):
            self._load_enrolled()

    def _set_phrase(self, index):
        self.phrase_index = index % len(PHRASES)
        phrase = PHRASES[self.phrase_index]
        counter_text = f"phrase {self.phrase_index + 1}/{len(PHRASES)}"
        # enroll deals its own words in the typing test, so only the two
        # phrase-at-a-time views follow the picker
        for capture, counter in ((self.identify_capture, self.identify_counter),
                                 (self.explain_capture, self.explain_counter)):
            capture.set_phrase(phrase)
            counter.configure(text=counter_text)

    def _step_phrase(self, delta):
        self._set_phrase(self.phrase_index + delta)

    def _random_phrase(self):
        if len(PHRASES) > 1:
            choices = [i for i in range(len(PHRASES)) if i != self.phrase_index]
            self._set_phrase(random.choice(choices))

    # ------------------------------------------------------------------
    # Key routing
    # ------------------------------------------------------------------

    def _route_keypress(self, event):
        if isinstance(self.root.focus_get(), tk.Entry):
            return None  # the user id field owns the keyboard while focused
        # the verdict travels back out: a capture widget returns "break" for a
        # key Tk would otherwise act on itself (Tab moving the focus mid-test)
        return self._capture_for(self.view).handle_keypress(event)

    def _route_keyrelease(self, event):
        if isinstance(self.root.focus_get(), tk.Entry):
            return None
        return self._capture_for(self.view).handle_keyrelease(event)

    # ------------------------------------------------------------------
    # Enroll view behaviour
    # ------------------------------------------------------------------

    def _status(self, label, text, colour):
        """Set a status line and fade the new text in, so a message that
        replaces another one is noticed rather than silently swapped."""
        label.configure(text=text)
        self.anim.tween(
            240,
            lambda p: label.configure(fg=blend(BG, colour, p)),
            key=f"status-{id(label)}")

    def _on_typing_test_finish(self, samples, summary):
        """Bank a finished typing test as enrollment samples.

        One run arrives as several samples -- the test is cut into windows of
        the length the pipeline reads -- so the count can cross the suggested
        enrollment size in a single go, and the prompt has to test for the
        crossing rather than for equality.
        """
        if not samples:
            self._status(self.enroll_status,
                         "too short to use - a sample needs at least "
                         f"{MIN_KEYS} keystrokes", FG_INCORRECT)
            self.enroll_capture.set_result_note("nothing recorded")
            return

        before = len(self.samples)
        for sample in samples:
            self.samples.append(sample)
            self._append_sample_line(len(self.samples), sample,
                                     sample_stats(sample))

        stats = sample_stats(samples[-1])
        self.signal_strip.show(stats)
        self._reveal_sample_stats(stats)

        plural = "" if len(samples) == 1 else "s"
        self._status(self.enroll_status,
                     f"{len(samples)} sample{plural} recorded at "
                     f"{summary['wpm']:.0f} wpm", OK)
        self.enroll_capture.set_result_note(
            f"+{len(samples)} sample{plural} added to this session")
        self._refresh_enroll_counts()

        if before < MIN_SAMPLES_SUGGESTED <= len(self.samples):
            messagebox.showinfo(
                "Good enrollment size",
                f"{len(self.samples)} samples recorded - past the "
                f"{MIN_SAMPLES_SUGGESTED} that make a stable template. Run "
                "another test for a tighter one, or save now.")

    SAMPLE_HEADER = "   #  keys   time   dwell  flight  typed\n"

    def _reveal_sample_stats(self, stats):
        """Count the headline numbers up from zero as the bars grow."""
        self.anim.tween(
            460,
            lambda p: self.sample_stats_label.configure(
                text=(f"{stats['chars']} keys   {stats['seconds'] * p:.2f} s   "
                      f"{stats['wpm'] * p:.0f} wpm\n"
                      f"mean dwell {stats['mean_dwell'] * p:.0f} ms   "
                      f"mean flight {stats['mean_flight'] * p:.0f} ms"),
                fg=blend(BG_PANEL, FG_DIM, min(1.0, p * 2))),
            key="sample-stats")

    def _append_sample_line(self, n, sample, stats):
        # newest first, under a fixed header; columns are sized to fit the
        # panel without horizontal scrolling
        phrase = " ".join(sample["phrase"].split())
        if len(phrase) > 15:
            phrase = phrase[:14] + "…"
        line = (f"  {n:>2}  {stats['chars']:>4}  {stats['seconds']:>5.2f}s  "
                f"{stats['mean_dwell']:>4.0f}ms  {stats['mean_flight']:>4.0f}ms  "
                f"{phrase}\n")
        # the new row lands in accent and cools to the list colour, so it is
        # obvious which line just arrived
        tag = f"row{n}"
        self.samples_text.tag_configure(tag, foreground=ACCENT)
        self.samples_text.config(state="normal")
        if not self.samples_text.get("1.0", "2.0").strip():
            self.samples_text.insert("1.0", self.SAMPLE_HEADER, "dim")
        self.samples_text.insert("2.0", line, tag)
        self.samples_text.config(state="disabled")

        def cool(p):
            self.samples_text.tag_configure(
                tag, foreground=blend(ACCENT, FG_LABEL, p))

        self.anim.tween(900, cool, key=tag)

    def _refresh_enroll_counts(self):
        n = len(self.samples)
        suffix = ("" if n >= MIN_SAMPLES_SUGGESTED
                  else f" / {MIN_SAMPLES_SUGGESTED} suggested")
        self.enroll_count_label.configure(
            text=f"{n} sample{'s' if n != 1 else ''} this session{suffix}")
        self.save_btn.set_enabled(n > 0)

    def _refresh_dir_label(self):
        self.dir_label.configure(text=f"enrollment folder: {self.data_dir}")

    def _choose_data_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.data_dir,
                                         title="Enrollment folder")
        if chosen:
            self.data_dir = chosen
            self._refresh_dir_label()
            self.enrolled = None
            self.threshold = None
            self._status(self.enrolled_label, "enrolled set not loaded",
                         FG_LABEL)
            if self.view in ("identify", "explain"):
                self._load_enrolled()

    def _current_user_id(self):
        user_id = self.user_entry.get().strip().replace(" ", "_")
        if not user_id:
            messagebox.showwarning("User id needed",
                                   "Enter a user id before saving samples.")
            return None
        return user_id

    def _record_for(self, user_id):
        # The record-level phrase is a label, not data: every sample carries
        # the text it was typed from, and no stage downstream reads either --
        # the features are timing. A session of typing tests has no single
        # phrase to name, so it says so.
        return {
            "user_id": user_id,
            "phrase": FREE_TEXT_LABEL,
            "samples": self.samples,
        }

    def _save_enrollment(self):
        if not self.samples:
            return
        user_id = self._current_user_id()
        if not user_id:
            return

        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, f"{user_id}.json")

        record = self._record_for(user_id)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                messagebox.showerror("Could not read existing file",
                                     f"{path}\n\n{exc}")
                return
            existing_n = len(existing.get("samples", []))
            if not messagebox.askyesno(
                "Append to existing enrollment?",
                f"{os.path.basename(path)} already holds {existing_n} sample(s) "
                f"for '{existing.get('user_id', '?')}'.\n\n"
                f"Append this session's {len(self.samples)} sample(s)?"):
                return
            record["samples"] = existing.get("samples", []) + self.samples

        with open(path, "w") as f:
            json.dump(record, f, indent=2)

        messagebox.showinfo(
            "Enrolled",
            f"{len(self.samples)} sample(s) saved to:\n{path}\n\n"
            f"'{user_id}' now has {len(record['samples'])} enrolled sample(s).")
        self._clear_session(confirm=False)
        self.enrolled = None  # force a rebuild next time identify runs

    def _export_json(self):
        if not self.samples:
            messagebox.showinfo("Nothing to export",
                                "Record at least one sample first.")
            return
        user_id = self._current_user_id()
        if not user_id:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile=f"{user_id}.json",
            filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, "w") as f:
            json.dump(self._record_for(user_id), f, indent=2)
        messagebox.showinfo("Saved", f"Samples written to:\n{path}")

    def _clear_session(self, confirm=True):
        if confirm and self.samples and not messagebox.askyesno(
                "Clear session",
                f"Discard {len(self.samples)} unsaved sample(s)?"):
            return
        self.samples = []
        self.samples_text.config(state="normal")
        self.samples_text.delete("1.0", "end")
        self.samples_text.config(state="disabled")
        self.signal_strip.clear()
        self.sample_stats_label.configure(text="")
        self.enroll_capture.reset()
        self._refresh_enroll_counts()

    # ------------------------------------------------------------------
    # Identify view behaviour
    # ------------------------------------------------------------------

    # Why an attempt was thrown away. Only the phrase-at-a-time views can
    # reject one: the typing test scores what was typed rather than demanding
    # an exact match, so a typo there is data like any other.
    REJECT_MESSAGES = {
        "mismatch": "that did not match the phrase exactly - retyped from scratch",
        "untracked": "could not track that input reliably - retyped from scratch",
    }

    def _on_identify_progress(self, pos, total):
        if pos == 0:
            self._status(self.identify_status,
                         "type the phrase to identify the typist", FG_DIM)
        else:
            self.identify_status.configure(text=f"{pos}/{total} characters",
                                           fg=FG_LABEL)

    def _on_identify_sample(self, sample, status):
        if sample is None:
            self._status(self.identify_status,
                         self.REJECT_MESSAGES.get(status, "attempt discarded"),
                         FG_INCORRECT)
            self.identify_capture.flash_reject()
            return
        if self.enrolled is None:
            self._status(self.identify_status,
                         "enrolled set still loading - try again in a moment",
                         WARN)
            if not self.loading:
                self._load_enrolled()
            return

        self._status(self.identify_status,
                     "matching against enrolled users...", FG_LABEL)
        self._set_verdict("...", "running the pipeline", FG_LABEL)
        self.attempt_stats = sample_stats(sample)
        self._submit_job("identify", self._job_identify, [sample])

    def _identify_from_file(self):
        path = filedialog.askopenfilename(
            initialdir=self.data_dir, title="Unknown sample JSON",
            filetypes=[("JSON", "*.json")])
        if not path:
            return
        if self.enrolled is None:
            self._load_enrolled()
            self._status(self.identify_status,
                         "enrolled set loading - pick the file again in a moment",
                         WARN)
            return
        self._set_verdict("...", f"identifying {os.path.basename(path)}",
                          FG_LABEL)
        self.attempt_stats = None  # a loaded file has no typed attempt to summarise
        self._submit_job("identify_file", self._job_identify_file, path)

    def _set_verdict(self, headline, sub, colour, detail=""):
        """Land the verdict by typing the name out, then fading the
        supporting lines in under it -- a keystroke project answering in
        keystrokes."""
        reveal_text(self.anim, self.verdict_label, headline,
                    duration=90 + 45 * len(headline), colour=colour,
                    from_colour=BG_PANEL, key="verdict")

        self.verdict_sub.configure(text=sub)
        self.verdict_detail.configure(text=detail)
        for label, target, delay in ((self.verdict_sub, FG_DIM, 220),
                                     (self.verdict_detail, FG_LABEL, 320)):
            self.anim.tween(
                260,
                lambda p, l=label, t=target: l.configure(
                    fg=blend(BG_PANEL, t, p)),
                delay=delay, key=f"verdict-{id(label)}")

    # ------------------------------------------------------------------
    # Explain view behaviour
    # ------------------------------------------------------------------

    def _on_explain_progress(self, pos, total):
        if pos == 0:
            self._status(self.explain_status,
                         "type the phrase to run the walkthrough", FG_DIM)
        else:
            self.explain_status.configure(text=f"{pos}/{total} characters",
                                          fg=FG_LABEL)

    def _on_explain_sample(self, sample, status):
        if sample is None:
            self._status(self.explain_status,
                         self.REJECT_MESSAGES.get(status, "attempt discarded"),
                         FG_INCORRECT)
            self.explain_capture.flash_reject()
            return
        if self.enrolled is None:
            self._status(self.explain_status,
                         "enrolled set still loading - try again in a moment",
                         WARN)
            if not self.loading:
                self._load_enrolled()
            return

        self._status(self.explain_status, "running the pipeline...", FG_LABEL)
        self._submit_job("explain", self._job_explain, sample)

    def _explain_retype(self):
        """Back to the typing card, for another sample."""
        self.explain_showing_walkthrough = False
        self.explain_walkthrough.pack_forget()
        self.explain_type_wrap.pack(fill="x", pady=(6, 20))
        self.explain_capture.active = self.view == "explain"
        self.explain_capture.reset()
        self._status(self.explain_status,
                     "type the phrase to run the walkthrough", FG_DIM)

    def _apply_explain(self, result):
        phases, summary = result
        self.explain_showing_walkthrough = True
        self.explain_capture.active = False
        self.explain_type_wrap.pack_forget()
        self.explain_walkthrough.pack(fill="both", expand=True)
        self.explain_walkthrough.load(phases, summary)
        self._status(self.explain_status,
                     f"walkthrough ready - this sample resolves to "
                     f"{summary['decision']}",
                     OK if summary["accepted"] else FG_INCORRECT)

    # ------------------------------------------------------------------
    # Worker thread plumbing
    # ------------------------------------------------------------------

    def _import_pipeline(self):
        """The pipeline handles, imported on first use (or by the splash)."""
        if self.pipeline is None:
            self.pipeline = load_pipeline()
        return self.pipeline

    def _submit_job(self, kind, fn, payload=None):
        self.loading = True
        self.spinner.start(side="left", padx=(0, 8))

        def run():
            try:
                result = fn(payload)
                self.results_queue.put((kind, result, None))
            except Exception as exc:  # surfaced in the UI, not swallowed
                self.results_queue.put((kind, None, exc))

        threading.Thread(target=run, daemon=True).start()

    def _load_enrolled(self, _payload=None):
        self._status(self.enrolled_label, "loading enrolled set...", WARN)
        self._submit_job("enrolled", self._job_load_enrolled)

    def _job_load_enrolled(self, _payload):
        p = self._import_pipeline()
        X, y, user_ids = p["build_dataset"](self.data_dir)
        threshold, eer = p["compute_open_set_threshold"](X, y, user_ids)
        return {"X": X, "y": y, "user_ids": user_ids,
                "threshold": threshold, "eer": eer}

    def _job_identify(self, samples):
        p = self._import_pipeline()
        X, y, _ = self.enrolled
        vecs = []
        for sample in samples:
            sig = p["sample_to_signal"](sample)
            if len(sig.dwell) == 0:
                continue
            vecs.append(p["signal_feature_vector"](sig))
        if not vecs:
            raise ValueError("sample had no usable keystrokes")
        unknown_X = p["np"].array(vecs)
        return p["identify"](unknown_X, X, y, k=3, threshold=self.threshold)

    def _job_identify_file(self, path):
        p = self._import_pipeline()
        X, y, _ = self.enrolled
        unknown_X = p["load_unknown_samples"](path)
        if len(unknown_X) == 0:
            raise ValueError(f"no usable samples in {os.path.basename(path)}")
        return p["identify"](unknown_X, X, y, k=3, threshold=self.threshold)

    def _job_explain(self, sample):
        p = self._import_pipeline()
        X, y, user_ids = self.enrolled
        enrolled = {"X": X, "y": y, "user_ids": user_ids,
                    "threshold": self.threshold, "eer": self.eer}
        return build_phases(sample, enrolled, p)

    def _poll_worker(self):
        try:
            while True:
                kind, result, error = self.results_queue.get_nowait()
                self.loading = False
                self.spinner.stop()
                if error is not None:
                    self._handle_job_error(kind, error)
                elif kind == "enrolled":
                    self._apply_enrolled(result)
                elif kind == "explain":
                    self._apply_explain(result)
                else:
                    self._apply_identification(result)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_worker)

    def _handle_job_error(self, kind, error):
        if kind == "explain":
            self._status(self.explain_status, str(error), FG_INCORRECT)
            return
        if kind == "enrolled":
            self._status(self.enrolled_label,
                         f"could not load enrolled set: {error}", FG_INCORRECT)
        else:
            self._set_verdict("error", str(error), FG_INCORRECT)
        self._status(self.identify_status, str(error), FG_INCORRECT)

    def _apply_enrolled(self, result):
        self.enrolled = (result["X"], result["y"], result["user_ids"])
        self.threshold = result["threshold"]
        self.eer = result["eer"]
        n_users = len(result["user_ids"])
        self._status(
            self.enrolled_label,
            (f"{len(result['X'])} samples  |  {n_users} users: "
             f"{', '.join(result['user_ids'][:6])}"
             f"{' ...' if n_users > 6 else ''}  |  "
             f"threshold {result['threshold']:.2f} "
             f"(EER {result['eer'] * 100:.1f}%)"),
            FG_LABEL)
        self._status(self.identify_status,
                     "ready - type the phrase to identify the typist", FG_DIM)

    def _apply_identification(self, results):
        p = self.pipeline
        unrecognized = p["UNRECOGNIZED"]

        # One verdict from every sample of the run, by pooling distances
        # rather than voting on per-sample answers -- identify_sample.fuse
        # explains why, and the app and the CLI now agree by construction.
        first = p["fuse"](results, threshold=self.threshold)
        overall = first["decision"]

        accepted = overall != unrecognized
        colour = ACCENT if accepted else FG_INCORRECT
        headline = overall if accepted else "unrecognized"

        if accepted:
            sub = (f"closest template: {first['closest_template']} at distance "
                   f"{first['closest_dist']:.2f} (threshold {self.threshold:.2f})")
        else:
            sub = (f"closest match {first['closest_template']} was still "
                   f"{first['closest_dist']:.2f} away, beyond the "
                   f"{self.threshold:.2f} accept threshold - likely not an "
                   f"enrolled typist")

        confidence = ("high" if first["margin"] > 1.0 else
                      "moderate" if first["margin"] > 0.4 else "low")
        detail = (f"knn vote (k=3)    {first['knn_vote']}\n"
                  f"margin over 2nd   {first['margin']:.2f}  ({confidence})\n"
                  f"samples scored    {len(results)}")
        if self.attempt_stats:
            a = self.attempt_stats
            detail += (f"\n\nthis attempt\n"
                       f"  mean dwell      {a['mean_dwell']:.0f} ms\n"
                       f"  mean flight     {a['mean_flight']:.0f} ms\n"
                       f"  speed           {a['wpm']:.0f} wpm")

        self._set_verdict(headline, sub, colour, detail)
        self.distance_bars.show(first["ranked_distances"], self.threshold,
                                first["closest_template"] if accepted else None)
        self._status(self.identify_status, "identification complete",
                     OK if accepted else FG_INCORRECT)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Keystroke behaviour capture and identification.")
    parser.add_argument(
        "--no-splash", dest="splash", action="store_false",
        help="skip the launch splash screen (shown by default; handy for "
             "tests and quick restarts)")
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help="folder of enrolled user JSON files (default: %(default)s)")
    parser.set_defaults(splash=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    root = tk.Tk()
    init_fonts(root)

    def start(preloaded=None):
        CaptureToolApp(root, preloaded=preloaded, data_dir=args.data_dir)
        root.deiconify()
        root.lift()

    if args.splash:
        from keyloggd.ui.splash import SplashScreen

        # the main window stays hidden until the splash has faded out, so the
        # two never overlap on screen
        root.withdraw()
        SplashScreen(root, args.data_dir, on_finish=start)
    else:
        start()

    root.mainloop()


if __name__ == "__main__":
    main()
