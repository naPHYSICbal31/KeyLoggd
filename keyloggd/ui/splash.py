"""
splash.py

Launch splash for the app, in the same Claude-inspired dark theme.

It is not just a curtain: the pause is spent doing the startup work that would
otherwise stall the first identification -- importing numpy, importing the
pipeline modules, building the enrolled feature matrix and its EER threshold
-- on a worker thread, with the progress bar reporting real stages rather than
a decorative crawl. Whatever it manages to load is handed to the app, so the
identify view is warm the moment the window opens.

The artwork is the project's own motif: a dwell/flight bar pattern that draws
itself left to right, the same shape the pipeline actually consumes.

Shown by default; `python main.py --no-splash` skips it.
"""

import queue
import random
import threading
import tkinter as tk

from keyloggd.ui.anim import Animator, blend, ease_out_cubic
from keyloggd.ui.theme import (ACCENT, ACCENT_DIM, BG, BG_PANEL, FG_DIM, FG_LABEL,
                               FG_TEXT, OK, ui)

WIDTH, HEIGHT = 540, 320
MIN_VISIBLE_MS = 1400      # a splash that flashes past is just a flicker
FADE_IN_MS = 260
FADE_OUT_MS = 220
BAR_COUNT = 28


class SplashScreen:
    """Borderless centred card that fades in, reports load progress, fades out.

    `on_finish(payload)` is called once, on the Tk main thread, after the fade
    out. payload is {"pipeline": ..., "enrolled": ...}; either value may be
    None if that stage failed, in which case the app just loads it lazily as
    it always did.
    """

    def __init__(self, root, data_dir, *, on_finish, min_visible_ms=MIN_VISIBLE_MS):
        self.root = root
        self.data_dir = data_dir
        self.on_finish = on_finish
        self.min_visible_ms = min_visible_ms

        self.results = queue.Queue()
        self.payload = {"pipeline": None, "enrolled": None}
        self.stage_index = 0
        self.progress = 0.0
        self.reveal = 0.0          # bar-pattern draw-in progress
        self.min_time_passed = False
        self.work_done = False
        self.finished = False

        # a fixed pattern rather than fresh randomness each launch, so the
        # splash looks like the same product every time it opens
        rng = random.Random(7)
        self.dwell_pattern = [0.35 + 0.65 * rng.random() for _ in range(BAR_COUNT)]
        self.flight_pattern = [0.25 + 0.55 * rng.random() for _ in range(BAR_COUNT)]

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.configure(bg=BG)
        self._centre()
        try:
            self.win.attributes("-topmost", True)
            self.win.attributes("-alpha", 0.0)
        except tk.TclError:
            pass

        self.anim = Animator(self.win)
        self._build()
        self._start_fade_in()
        self._start_work()
        self.root.after(self.min_visible_ms, self._min_time_elapsed)
        self.root.after(60, self._poll)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _centre(self):
        screen_w = self.win.winfo_screenwidth()
        screen_h = self.win.winfo_screenheight()
        x = (screen_w - WIDTH) // 2
        y = (screen_h - HEIGHT) // 3   # a third down reads better than centred
        self.win.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")

    def _build(self):
        # hairline accent border: an overrideredirect window has no frame of
        # its own, so the card needs its own edge to sit on a dark desktop
        border = tk.Frame(self.win, bg=ACCENT_DIM)
        border.pack(fill="both", expand=True)
        card = tk.Frame(border, bg=BG_PANEL)
        card.pack(fill="both", expand=True, padx=1, pady=1)

        tk.Label(card, text="keyloggd", fg=ACCENT, bg=BG_PANEL,
                 font=ui(38, bold=True)).pack(pady=(46, 0))
        tk.Label(card, text="keystroke behaviour capture & identification",
                 fg=FG_LABEL, bg=BG_PANEL, font=ui(11)).pack(pady=(4, 0))

        self.canvas = tk.Canvas(card, height=86, bg=BG_PANEL, bd=0,
                                highlightthickness=0)
        self.canvas.pack(fill="x", padx=44, pady=(26, 0))
        self.canvas.bind("<Configure>", lambda e: self._draw_bars())

        self.stage_label = tk.Label(card, text="", fg=FG_DIM, bg=BG_PANEL,
                                    font=ui(10))
        self.stage_label.pack(pady=(14, 0))

        self.bar = tk.Canvas(card, height=3, bg=BG_PANEL, bd=0,
                             highlightthickness=0)
        self.bar.pack(fill="x", padx=44, pady=(10, 0))
        self.bar.bind("<Configure>", lambda e: self._draw_progress())

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_bars(self):
        """The dwell/flight motif, mirrored about a centre line, drawing in
        left to right as the load progresses."""
        canvas = self.canvas
        canvas.delete("all")
        w = max(canvas.winfo_width(), 1)
        h = max(canvas.winfo_height(), 1)
        mid = h // 2
        step = w / BAR_COUNT
        bar_w = max(2, min(9, step - 4))

        for i in range(BAR_COUNT):
            # each bar has its own slice of the reveal, so the pattern sweeps
            local = min(1.0, max(0.0, self.reveal * BAR_COUNT - i))
            if local <= 0:
                continue
            x = i * step + (step - bar_w) / 2
            up = self.dwell_pattern[i] * (mid - 6) * local
            down = self.flight_pattern[i] * (mid - 6) * local
            canvas.create_rectangle(x, mid - up, x + bar_w, mid - 2,
                                    fill=blend(BG_PANEL, ACCENT, 0.35 + 0.65 * local),
                                    outline="")
            canvas.create_rectangle(x, mid + 2, x + bar_w, mid + down,
                                    fill=blend(BG_PANEL, FG_DIM, 0.4 + 0.6 * local),
                                    outline="")

        canvas.create_line(0, mid, w * min(1.0, self.reveal * 1.15), mid,
                           fill=blend(BG_PANEL, ACCENT_DIM, 0.9))

    def _draw_progress(self):
        bar = self.bar
        bar.delete("all")
        w = max(bar.winfo_width(), 1)
        h = max(bar.winfo_height(), 1)
        bar.create_rectangle(0, h - 2, w, h, fill=blend(BG_PANEL, BG, 0.8),
                             outline="")
        if self.progress > 0:
            bar.create_rectangle(0, h - 3, max(2, w * self.progress), h,
                                 fill=ACCENT, outline="")

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _start_fade_in(self):
        def frame(p):
            try:
                self.win.attributes("-alpha", p)
            except tk.TclError:
                pass

        self.anim.tween(FADE_IN_MS, frame, easing=ease_out_cubic, key="fade")

        def bars(p):
            self.reveal = p
            self._draw_bars()

        self.anim.tween(1100, bars, easing=ease_out_cubic, key="bars")

    def _animate_progress(self, target):
        start = self.progress

        def frame(p):
            self.progress = start + (target - start) * p
            self._draw_progress()

        self.anim.tween(300, frame, easing=ease_out_cubic, key="progress")

    def _set_stage(self, text, colour=FG_DIM):
        self.stage_label.configure(text=text)
        self.anim.tween(
            200,
            lambda p: self.stage_label.configure(fg=blend(BG_PANEL, colour, p)),
            key="stage")

    # ------------------------------------------------------------------
    # Background load
    # ------------------------------------------------------------------

    STAGES = ("warming up", "loading the pipeline", "reading the enrolled set",
              "ready")

    def _stage_text(self, index):
        """The last stage reports what was actually loaded, which is more use
        than the word 'ready' and confirms the app found the right folder."""
        enrolled = self.payload.get("enrolled")
        if index == len(self.STAGES) - 1 and enrolled:
            users = len(enrolled["user_ids"])
            samples = len(enrolled["X"])
            return (f"{users} user{'s' if users != 1 else ''}, "
                    f"{samples} samples enrolled")
        return self.STAGES[index]

    def _start_work(self):
        self._set_stage(self.STAGES[0])

        def run():
            try:
                from keyloggd.ui.app import load_pipeline
                self.results.put(("stage", 1, None))
                pipeline = load_pipeline()
                self.payload["pipeline"] = pipeline
                self.results.put(("stage", 2, None))

                X, y, user_ids = pipeline["build_dataset"](self.data_dir)
                threshold, eer = pipeline["compute_open_set_threshold"](
                    X, y, user_ids)
                self.payload["enrolled"] = {
                    "X": X, "y": y, "user_ids": user_ids,
                    "threshold": threshold, "eer": eer,
                }
                self.results.put(("stage", 3, None))
            except Exception as exc:
                # a missing/empty data dir is a normal first run: the app will
                # say so in the identify view, the splash should not block
                self.results.put(("error", 3, exc))
            self.results.put(("done", None, None))

        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, index, error = self.results.get_nowait()
                if kind == "stage":
                    self.stage_index = index
                    self._set_stage(self._stage_text(index))
                    self._animate_progress(index / (len(self.STAGES) - 1))
                elif kind == "error":
                    self._set_stage(f"no enrolled set yet ({type(error).__name__})",
                                    FG_TEXT)
                    self._animate_progress(1.0)
                elif kind == "done":
                    self.work_done = True
        except queue.Empty:
            pass

        if self.work_done and self.min_time_passed:
            self._finish()
            return
        self.root.after(60, self._poll)

    def _min_time_elapsed(self):
        self.min_time_passed = True

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _finish(self):
        if self.finished:
            return
        self.finished = True
        if self.payload["enrolled"]:
            self._set_stage(self._stage_text(len(self.STAGES) - 1), OK)

        def frame(p):
            try:
                self.win.attributes("-alpha", 1.0 - p)
            except tk.TclError:
                pass

        def done():
            self.anim.cancel_all()
            try:
                self.win.destroy()
            except tk.TclError:
                pass
            self.on_finish(self.payload)

        self.anim.tween(FADE_OUT_MS, frame, on_done=done, key="fade")
