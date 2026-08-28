"""
explain.py

The "how does it decide?" walkthrough: type a phrase, then step through what
the pipeline actually does to it, one phase per screen, with the real numbers
from the real modules at every step.

Nothing here re-implements the maths. Each phase calls the same functions the
identify path calls -- signal_construction.events_to_signal, fft_features
(zero_pad / compute_spectrum / spectral_features / feature_vector), classifier
(ZScoreScaler / build_templates / knn_predict) -- and simply keeps the
intermediate values so they can be plotted. If the pipeline changes, this view
changes with it rather than drifting into a pretty lie.

Phases, and the chart form each one earns:

  1 capture    dwell/flight per keystroke      grouped columns, two series (ms)
  2 pad        zero-padding to N = 32          the same columns + a shaded band
  3 spectrum   |FFT| per frequency bin         grouped columns, two series
  4 features   the 14-number feature vector    a table, not a chart: the
                                               components have different units,
                                               and one axis cannot honestly
                                               carry centroids, ratios and
                                               milliseconds at once
  5 normalise  z-scores vs the enrolled set    diverging columns about zero
  6 decide     distance to every template      horizontal bars + threshold
"""

import tkinter as tk
from dataclasses import dataclass, field

from anim import Animator, blend, ease_out_cubic
from charts import (SERIES_1, SERIES_2, Band, Chart, ChartSpec, RefLine,
                    Series)
from theme import (ACCENT, BG, BG_PANEL, FG_DIM, FG_INCORRECT, FG_LABEL,
                   FG_TEXT, OK, PillButton, mono, ui)

# (full name for the values panel, compact axis code). Fourteen categories
# share one axis, so the codes have to stay short enough not to collide --
# the panel beside the chart carries the full names.
FEATURE_NAMES = [
    ("dwell centroid", "dCen"),
    ("dwell low-band energy", "dLow"),
    ("dwell mid-band energy", "dMid"),
    ("dwell high-band energy", "dHi"),
    ("dwell total energy", "dE"),
    ("flight centroid", "fCen"),
    ("flight low-band energy", "fLow"),
    ("flight mid-band energy", "fMid"),
    ("flight high-band energy", "fHi"),
    ("flight total energy", "fE"),
    ("mean dwell", "mDw"),
    ("std dwell", "sdDw"),
    ("mean flight", "mFl"),
    ("std flight", "sdFl"),
]

AUTOPLAY_MS = 3200


@dataclass
class Phase:
    key: str
    title: str
    lead: str                      # one sentence: what this phase does
    chart: object = None           # ChartSpec, or None when a table is right
    rows: list = field(default_factory=list)   # (name, value) for the panel
    rows_title: str = "values"
    footnote: str = ""


def build_phases(sample, enrolled, pipeline):
    """Run the pipeline on one captured sample, keeping every intermediate."""
    np = pipeline["np"]
    signal = pipeline["sample_to_signal"](sample)
    dwell, flight = signal.dwell, signal.flight
    if len(dwell) == 0:
        raise ValueError("that sample has no usable keystrokes")

    fft_len = pipeline["FFT_LEN"]
    zero_pad = pipeline["zero_pad"]
    padded_dwell = zero_pad(dwell, fft_len)
    padded_flight = zero_pad(flight, fft_len)

    dwell_spec = pipeline["compute_spectrum"](dwell, fft_len)
    flight_spec = pipeline["compute_spectrum"](flight, fft_len)
    dwell_feat = pipeline["spectral_features"](dwell_spec)
    flight_feat = pipeline["spectral_features"](flight_spec)

    vector = pipeline["feature_vector"](dwell, flight)

    X, y, user_ids = enrolled["X"], enrolled["y"], enrolled["user_ids"]
    scaler = pipeline["ZScoreScaler"]().fit(X)
    z = scaler.transform(vector.reshape(1, -1))[0]
    templates = pipeline["build_templates"](X, y, scaler)
    vote = pipeline["knn_predict"](scaler.transform(X), y, z, k=3)

    distances = sorted(
        ((user, float(np.linalg.norm(z - templates[user]))) for user in user_ids),
        key=lambda kv: kv[1])
    closest_user, closest_dist = distances[0]
    runner_up = distances[1][1] if len(distances) > 1 else float("inf")
    threshold = enrolled["threshold"]
    accepted = closest_dist <= threshold
    decision = vote if accepted else pipeline["UNRECOGNIZED"]

    ms = lambda values: [float(v) * 1000.0 for v in values]

    phases = [
        Phase(
            key="capture",
            title="1 - capture",
            lead="Every keypress and release is timestamped, giving two "
                 "signals: how long each key is held (dwell) and the gap "
                 "between releasing one key and pressing the next (flight).",
            chart=ChartSpec(
                series=[Series(ms(dwell), "dwell", SERIES_1, "ms"),
                        Series(ms(flight), "flight", SERIES_2, "ms")],
                x_label="keystroke index", y_label="milliseconds",
                value_format="{:.0f}", tick_format="{:g}",
                hover_format="keystroke {index}",
                annotate_label=f"{max(ms(dwell)):.0f} ms"),
            rows=[("keystrokes", f"{len(dwell)}"),
                  ("mean dwell", f"{np.mean(dwell) * 1000:.1f} ms"),
                  ("std dwell", f"{np.std(dwell) * 1000:.1f} ms"),
                  ("mean flight", f"{np.mean(flight) * 1000:.1f} ms"),
                  ("std flight", f"{np.std(flight) * 1000:.1f} ms"),
                  ("total time", f"{sample['events'][-1]['t']:.2f} s")],
            rows_title="the raw signal",
            footnote="signal_construction.events_to_signal",
        ),
        Phase(
            key="pad",
            title="2 - zero-pad",
            lead=f"Both signals are padded with zeros to a fixed length of "
                 f"{fft_len}, so samples of different lengths land on the same "
                 f"frequency grid and stay comparable.",
            chart=ChartSpec(
                series=[Series(ms(padded_dwell), "dwell", SERIES_1, "ms"),
                        Series(ms(padded_flight), "flight", SERIES_2, "ms")],
                x_label=f"sample index (padded to N = {fft_len})",
                y_label="milliseconds", value_format="{:.0f}",
                bands=[Band(len(dwell), fft_len, "zeros")],
                hover_format="sample {index}", annotate_max=False),
            rows=[("dwell length", f"{len(dwell)} -> {fft_len}"),
                  ("flight length", f"{len(flight)} -> {fft_len}"),
                  ("padding added",
                   f"{fft_len - len(dwell)} / {fft_len - len(flight)}")],
            rows_title="lengths",
            footnote="fft_features.zero_pad",
        ),
        Phase(
            key="spectrum",
            title="3 - FFT",
            lead="The FFT turns each signal into a magnitude spectrum: how "
                 "typing energy is spread across rhythm frequencies. This is "
                 "steadier run to run than the raw timings, where one slow "
                 "keystroke shifts everything after it.",
            chart=ChartSpec(
                series=[Series([float(v) for v in dwell_feat["magnitude"]],
                               "|dwell|", SERIES_1),
                        Series([float(v) for v in flight_feat["magnitude"]],
                               "|flight|", SERIES_2)],
                x_label="frequency bin (0 = DC, the mean)",
                y_label="magnitude", value_format="{:.2f}",
                tick_format="{:g}", hover_format="bin {index}",
                # point at the dominant bin the summary quotes, not at DC:
                # bin 0 is just the signal's mean and is always the tallest
                annotate_index=int(dwell_feat["dominant_bin"]),
                annotate_label=f"dominant bin {dwell_feat['dominant_bin']}"),
            rows=[("dominant bin (dwell)", f"{dwell_feat['dominant_bin']}"),
                  ("dominant bin (flight)", f"{flight_feat['dominant_bin']}"),
                  ("dwell centroid", f"{dwell_feat['centroid']:.3f}"),
                  ("flight centroid", f"{flight_feat['centroid']:.3f}"),
                  ("bins kept", f"{len(dwell_feat['magnitude'])} of {fft_len}")],
            rows_title="spectrum",
            footnote="fft_features.compute_spectrum / spectral_features",
        ),
        Phase(
            key="features",
            title="4 - feature vector",
            lead="Each spectrum is summarised into five numbers, and four "
                 "plain time-domain statistics are added: 14 numbers that "
                 "stand in for this sample from here on. They are deliberately "
                 "not plotted together -- centroids, energy ratios and "
                 "milliseconds share no axis.",
            rows=[(FEATURE_NAMES[i][0], f"{vector[i]:.4f}")
                  for i in range(len(vector))],
            rows_title="the 14 features",
            footnote="fft_features.feature_vector",
        ),
        Phase(
            key="normalise",
            title="5 - normalise",
            lead="Those 14 numbers have wildly different scales, so each is "
                 "z-scored against the enrolled set: zero means exactly "
                 "average for the enrolled population, +/-1 means one standard "
                 "deviation away. Only now can they share one axis.",
            chart=ChartSpec(
                kind="diverging",
                series=[Series([float(v) for v in z], "z-score", SERIES_1)],
                categories=[short for _, short in FEATURE_NAMES],
                x_label="feature", y_label="standard deviations from the mean",
                value_format="{:+.2f}", tick_format="{:g}",
                hover_format="{index}",
                annotate_label=f"{z[int(np.argmax(np.abs(z)))]:+.2f}"),
            rows=[(FEATURE_NAMES[i][0], f"{z[i]:+.2f}")
                  for i in range(len(z))],
            rows_title="z-scores",
            footnote="classifier.ZScoreScaler",
        ),
        Phase(
            key="decide",
            title="6 - decide",
            lead=f"Each enrolled user has a template: the mean of their own "
                 f"normalised samples. The sample is compared to every "
                 f"template, and the nearest wins -- but only if it is inside "
                 f"the accept threshold, otherwise nobody is claimed.",
            chart=ChartSpec(
                kind="hbars",
                series=[Series([d for _, d in distances], "distance")],
                categories=[u for u, _ in distances],
                highlight=0 if accepted else None,
                value_format="{:.2f}",
                reflines=[RefLine(threshold, f"accept threshold "
                                             f"{threshold:.2f}")]),
            rows=[("decision", str(decision)),
                  ("kNN vote (k=3)", str(vote)),
                  ("closest template", f"{closest_user}"),
                  ("distance", f"{closest_dist:.2f}"),
                  ("threshold", f"{threshold:.2f}"),
                  ("margin over 2nd", f"{runner_up - closest_dist:.2f}"),
                  ("enrolled users", f"{len(user_ids)}")],
            rows_title="verdict",
            footnote="classifier.build_templates / knn_predict, "
                     "identify_sample.identify",
        ),
    ]
    summary = {"decision": decision, "accepted": accepted, "vote": vote,
               "closest": closest_user, "distance": closest_dist,
               "threshold": threshold}
    return phases, summary


class ExplainView(tk.Frame):
    """Phase rail + chart + values panel, driven by build_phases()."""

    def __init__(self, parent, *, on_replay):
        super().__init__(parent, bg=BG)
        self.on_replay = on_replay
        self.phases = []
        self.summary = None
        self.index = 0
        self.playing = False
        self.play_job = None
        self.anim = Animator(self)

        rail = tk.Frame(self, bg=BG)
        rail.pack(fill="x")
        self.rail = rail
        self.chips = []

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, pady=(14, 0))
        body.columnconfigure(0, weight=5, uniform="explain")
        body.columnconfigure(1, weight=2, uniform="explain")
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG_PANEL)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.title_label = tk.Label(left, text="", fg=ACCENT, bg=BG_PANEL,
                                    font=ui(17, bold=True), anchor="w")
        self.title_label.pack(fill="x", padx=18, pady=(16, 2))
        self.lead_label = tk.Label(left, text="", fg=FG_LABEL, bg=BG_PANEL,
                                   font=ui(11), anchor="w", justify="left",
                                   wraplength=520)
        self.lead_label.pack(fill="x", padx=18, pady=(0, 10))
        self.chart = Chart(left, height=250)
        self.chart.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        self.footnote = tk.Label(left, text="", fg=FG_DIM, bg=BG_PANEL,
                                 font=mono(9), anchor="w")
        self.footnote.pack(fill="x", padx=18, pady=(0, 12))

        right = tk.Frame(body, bg=BG_PANEL)
        right.grid(row=0, column=1, sticky="nsew")
        self.rows_title = tk.Label(right, text="", fg=FG_DIM, bg=BG_PANEL,
                                   font=ui(9), anchor="w")
        self.rows_title.pack(fill="x", padx=16, pady=(16, 6))
        self.rows_text = tk.Text(right, bg=BG_PANEL, fg=FG_LABEL, font=mono(10),
                                 bd=0, highlightthickness=0, padx=16, pady=0,
                                 wrap="none", state="disabled", cursor="arrow")
        self.rows_text.pack(fill="both", expand=True, pady=(0, 12))
        self.rows_text.tag_configure("name", foreground=FG_DIM)
        self.rows_text.tag_configure("value", foreground=FG_TEXT)
        self.rows_text.tag_configure("accept", foreground=OK)
        self.rows_text.tag_configure("reject", foreground=FG_INCORRECT)

        controls = tk.Frame(self, bg=BG)
        controls.pack(fill="x", pady=(14, 0))
        self.back_btn = PillButton(controls, "back", self.previous, size=10,
                                   bg=BG, padx=14, pady=6)
        self.back_btn.pack(side="left")
        self.next_btn = PillButton(controls, "next phase", self.next,
                                   kind="primary", size=10, bg=BG,
                                   padx=16, pady=6)
        self.next_btn.pack(side="left", padx=8)
        self.play_btn = PillButton(controls, "play all", self.toggle_play,
                                   size=10, bg=BG, padx=14, pady=6)
        self.play_btn.pack(side="left")
        PillButton(controls, "type another sample", self._replay, size=10,
                   bg=BG, padx=14, pady=6).pack(side="right")
        self.step_label = tk.Label(controls, text="", fg=FG_DIM, bg=BG,
                                   font=ui(10))
        self.step_label.pack(side="right", padx=14)

    # -- content ------------------------------------------------------------

    def load(self, phases, summary):
        self.phases = phases
        self.summary = summary
        self._build_rail()
        self.index = 0
        self._render(animate=True)

    def _build_rail(self):
        for chip in self.chips:
            chip.destroy()
        self.chips = []
        for i, phase in enumerate(self.phases):
            chip = tk.Label(self.rail, text=phase.title.replace(" - ", "  "),
                            fg=FG_DIM, bg=BG, font=ui(10), padx=10, pady=4,
                            cursor="hand2")
            chip.pack(side="left", padx=(0, 6))
            chip.bind("<Button-1>", lambda e, n=i: self.go(n))
            self.chips.append(chip)

    def _render(self, animate=True):
        if not self.phases:
            return
        phase = self.phases[self.index]
        self.title_label.configure(text=phase.title)
        self.lead_label.configure(text=phase.lead)
        self.footnote.configure(text=phase.footnote)
        self.rows_title.configure(text=phase.rows_title.upper())
        self.step_label.configure(
            text=f"phase {self.index + 1} of {len(self.phases)}")

        if phase.chart:
            self.chart.pack(fill="both", expand=True, padx=8, pady=(0, 6))
            self.chart.show(phase.chart, animate=animate)
        else:
            # phase 4 is a table on purpose; hiding the plot area gives the
            # numbers the whole panel rather than leaving an empty frame
            self.chart.clear()
            self.chart.pack_forget()

        self._fill_rows(phase)
        for i, chip in enumerate(self.chips):
            active = i == self.index
            chip.configure(fg=ACCENT if active else
                           (FG_LABEL if i < self.index else FG_DIM),
                           font=ui(10, bold=active))
        self.back_btn.set_enabled(self.index > 0)
        self.next_btn.set_enabled(self.index < len(self.phases) - 1)

    def _fill_rows(self, phase):
        self.rows_text.configure(state="normal")
        self.rows_text.delete("1.0", "end")
        width = max((len(name) for name, _ in phase.rows), default=0) + 2
        for name, value in phase.rows:
            tag = "value"
            if name == "decision" and self.summary:
                tag = "accept" if self.summary["accepted"] else "reject"
            self.rows_text.insert("end", f"{name:<{width}}", "name")
            self.rows_text.insert("end", f"{value}\n", tag)
        self.rows_text.configure(state="disabled")

    # -- navigation ---------------------------------------------------------

    def go(self, index, animate=True):
        if not self.phases:
            return
        self.index = max(0, min(len(self.phases) - 1, index))
        self._render(animate=animate)

    def next(self):
        if self.index < len(self.phases) - 1:
            self.go(self.index + 1)
        else:
            self.stop_play()

    def previous(self):
        self.go(self.index - 1)

    def toggle_play(self):
        if self.playing:
            self.stop_play()
        else:
            self.playing = True
            self.play_btn.set_text("pause")
            self._schedule_play()

    def stop_play(self):
        self.playing = False
        self.play_btn.set_text("play all")
        if self.play_job:
            self.after_cancel(self.play_job)
            self.play_job = None

    def _schedule_play(self):
        if not self.playing:
            return
        self.play_job = self.after(AUTOPLAY_MS, self._advance_play)

    def _advance_play(self):
        if not self.playing:
            return
        if self.index >= len(self.phases) - 1:
            self.stop_play()
            return
        self.go(self.index + 1)
        self._schedule_play()

    def _replay(self):
        self.stop_play()
        self.on_replay()
