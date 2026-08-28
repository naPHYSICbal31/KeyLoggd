"""
Keystroke Acoustic Annotation Platform + Classifier
====================================================

Two halves of one loop:

  1. ANNOTATE - hit record, then just type. The mic records continuously
     while every key you press is timestamped against the audio clock. Stop
     whenever you like; the session is then sliced into one short .wav per
     keystroke, filed under a folder named after the key that made it.
  2. TRAIN / PREDICT - those labelled folders feed the feature extractor and
     classifier further down this file.

So nobody hand-labels clips: the keyboard labels itself as you type.

    python train_keystroke_classifier.py                 # annotation platform
    python train_keystroke_classifier.py --train --data_dir ./dataset
    python train_keystroke_classifier.py --infer         # inference platform
    python train_keystroke_classifier.py --infer_file recording.wav
    python train_keystroke_classifier.py --predict clip.wav

HOW THE ALIGNMENT WORKS
-----------------------
Recording and typing run on different clocks, so the two are stitched
together in three steps:

  * The audio callback stores an anchor per block: (frame index, the wall
    time that block's first sample hit the ADC). PortAudio reports that ADC
    time, so the input latency of the mic path is already accounted for.
  * Each keypress is stamped with `time.perf_counter()` in the Tk handler
    and converted to an approximate sample index through the nearest anchor.
  * That estimate is then refined against the audio itself: inside a short
    search window around it, the loudest peak is found and walked back to
    the start of its attack. The few ms of jitter between the OS key event
    and the actual click disappear at this step.

Every clip is judged before it is filed, so doubtful audio is set aside
instead of quietly poisoning the training set:

    overlap   another key landed inside this clip's window
    repeat    key auto-repeated from being held down
    quiet     no clear click above the session's noise floor
    clipped   the input was driven into the rails
    unmapped  a key with no label (modifiers, function keys, ...)

WHAT A SESSION LEAVES ON DISK
-----------------------------
    dataset/
        a/  a__20260828-142530__0003.wav      <- training clips
        b/  ...
        space/ ...
        _sessions/
            20260828-142530.wav               <- the untouched recording
            20260828-142530.json              <- events, timings, verdicts
        _rejected/
            overlap/ ...                      <- only with --keep_flagged

Because the raw recording and every event time are kept, the slicing is
reproducible: change the window or the thresholds and re-cut old sessions
without retyping a thing.

    python train_keystroke_classifier.py --resegment ./dataset --post_ms 120

DATA COLLECTION NOTES
---------------------
- Keep mic, distance, keyboard and surface identical between the data you
  record and anything you later classify. Models trained on keystroke audio
  generalize badly across any change to that chain.
- Aim for 40+ clips per key. The coverage grid shows which keys are still
  thin, and can generate practice text out of the weakest ones.
- Type at a normal-ish pace but avoid real overlap: two keys within ~80ms
  share one sound, and both get dropped as `overlap`.

REQUIREMENTS
------------
    annotating:  numpy, sounddevice          (wav writing is stdlib)
    training:    numpy, librosa, scikit-learn, joblib
"""

import argparse
import bisect
import glob
import json
import os
import queue
import random
import string
import sys
import threading
import time
import warnings
import wave
from collections import deque
from datetime import datetime

import numpy as np

import tkinter as tk
from tkinter import filedialog

# The annotation UI shares the project's Claude-dark theme. An import failure
# is tolerated so a standalone copy of this file can still train and predict;
# only the UI needs it, and it says so when launched.
_THEME_ERROR = None
try:
    from theme import (ACCENT, BG, BG_INPUT, BG_PANEL, FG_DIM, FG_INCORRECT,
                       FG_LABEL, FG_TEXT, OK, WARN, PillButton, RoundedPanel,
                       Spinner, init_fonts, mono, rounded_rect_points, ui)
    from anim import blend
except Exception as exc:                                    # pragma: no cover
    _THEME_ERROR = exc

warnings.filterwarnings("ignore")

DEFAULT_DATA_DIR = "dataset"
SESSIONS_DIRNAME = "_sessions"
REJECTED_DIRNAME = "_rejected"
DEFAULT_SR = 16000
DEFAULT_TARGET = 40
MANIFEST_VERSION = 1


# --------------------------------------------------------------------------
# Key labelling
# --------------------------------------------------------------------------
#
# A label names the *physical key*, not the character produced: shift+a and a
# are the same key making the same sound, and on a case-insensitive
# filesystem "a/" and "A/" would be the same folder anyway. Shifted keysyms
# therefore fold back onto their unshifted key (US layout).

_LETTER_LABELS = {c: c for c in string.ascii_lowercase}
_LETTER_LABELS.update({c.upper(): c for c in string.ascii_lowercase})

_DIGIT_LABELS = {d: d for d in "0123456789"}
_SHIFTED_DIGITS = {
    "exclam": "1", "at": "2", "numbersign": "3", "dollar": "4",
    "percent": "5", "asciicircum": "6", "ampersand": "7", "asterisk": "8",
    "parenleft": "9", "parenright": "0",
}

_PUNCT_LABELS = {
    "minus": "minus", "underscore": "minus",
    "equal": "equal", "plus": "equal",
    "bracketleft": "bracketleft", "braceleft": "bracketleft",
    "bracketright": "bracketright", "braceright": "bracketright",
    "backslash": "backslash", "bar": "backslash",
    "semicolon": "semicolon", "colon": "semicolon",
    "apostrophe": "apostrophe", "quotedbl": "apostrophe",
    "grave": "grave", "asciitilde": "grave",
    "comma": "comma", "less": "comma",
    "period": "period", "greater": "period",
    "slash": "slash", "question": "slash",
}

_SPECIAL_LABELS = {
    "space": "space",
    "Return": "enter",
    "BackSpace": "backspace",
    "Tab": "tab",
    "KP_Enter": "kp_enter",
}
_SPECIAL_LABELS.update({f"KP_{d}": f"kp_{d}" for d in "0123456789"})

KEYSYM_LABELS = {}
for _src in (_LETTER_LABELS, _DIGIT_LABELS, _SHIFTED_DIGITS, _PUNCT_LABELS,
             _SPECIAL_LABELS):
    KEYSYM_LABELS.update(_src)

# Keys that are part of typing but have no keystroke of their own worth
# labelling, plus the app's own controls. Held modifiers also auto-repeat,
# which would otherwise flood a session with junk events.
IGNORED_KEYSYMS = {
    "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
    "Meta_L", "Meta_R", "Super_L", "Super_R", "Caps_Lock", "Num_Lock",
    "Scroll_Lock", "ISO_Level3_Shift", "Escape", "Win_L", "Win_R",
    "App", "Menu",
}

# The key set the coverage grid tracks, in a keyboard-ish reading order.
TRACKED_LABELS = (
    list("1234567890") + ["minus", "equal"]
    + list("qwertyuiop") + ["bracketleft", "bracketright", "backslash"]
    + list("asdfghjkl") + ["semicolon", "apostrophe"]
    + list("zxcvbnm") + ["comma", "period", "slash"]
    + ["space", "enter", "backspace", "tab", "grave"]
)

# Shown in the coverage grid instead of the folder name, which is too wide.
LABEL_GLYPHS = {
    "space": "spc", "enter": "ent", "backspace": "bsp", "tab": "tab",
    "minus": "-", "equal": "=", "bracketleft": "[", "bracketright": "]",
    "backslash": "\\", "semicolon": ";", "apostrophe": "'", "grave": "`",
    "comma": ",", "period": ".", "slash": "/",
}

# Typed into the prompt line to spread coverage over the whole keyboard.
PROMPTS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "how vexingly quick daft zebras jump",
    "sphinx of black quartz, judge my vow",
    "jackdaws love my big sphinx of quartz",
    "0123456789 - = [ ] \\ ; ' , . /",
    "waltz, bad nymph, for quick jigs vex",
    "the five boxing wizards jump quickly",
]


def keysym_to_label(keysym):
    """Folder-safe label for a Tk keysym, or None if it is not a labelled key.

    Returns None for both ignored keys (modifiers, escape) and keys absent
    from the map; callers that care about the difference test the keysym
    against IGNORED_KEYSYMS themselves.
    """
    if keysym in IGNORED_KEYSYMS:
        return None
    return KEYSYM_LABELS.get(keysym)


def label_display(label):
    """Short glyph for a label, for grids and summaries."""
    return LABEL_GLYPHS.get(label, label)


# --------------------------------------------------------------------------
# WAV I/O (stdlib only, so annotating needs no soundfile/librosa)
# --------------------------------------------------------------------------

def write_wav(path, y, sr):
    """Write mono float samples in [-1, 1] as 16-bit PCM."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def read_wav(path):
    """Read a 16-bit PCM wav back as (mono float32, sample_rate)."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        channels = w.getnchannels()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    y = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        y = y.reshape(-1, channels).mean(axis=1)
    return y, sr


# --------------------------------------------------------------------------
# Recording a session
# --------------------------------------------------------------------------

class SessionRecorder:
    """Continuous mic capture with a clock shared with the key handler.

    The audio thread only ever appends: blocks to `_blocks`, and one
    (frame index, ADC wall time) anchor per block to `_anchors`. Keypresses
    arrive on the Tk thread and are appended to `_events` with a
    `perf_counter()` stamp. Nothing is aligned until stop().
    """

    def __init__(self, sr=DEFAULT_SR, device=None, blocksize=1024):
        self.sr = int(sr)
        self.device = device
        self.blocksize = blocksize

        self._stream = None
        self._lock = threading.Lock()
        self._blocks = []
        self._anchors = []          # (frame_index, adc_perf_time)
        self._events = []           # dicts, see mark_key()
        self._frames = 0
        self._overflows = 0
        self._level = 0.0
        self._peak = 0.0
        self._t_start = None

    # -- lifecycle ----------------------------------------------------------

    @staticmethod
    def available():
        """True when sounddevice can be imported (PortAudio present)."""
        try:
            import sounddevice  # noqa: F401
        except Exception:
            return False
        return True

    def start(self):
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(
                "recording needs the 'sounddevice' package:\n"
                "    pip install sounddevice") from exc

        self._blocks, self._anchors, self._events = [], [], []
        self._frames = self._overflows = 0
        self._level = self._peak = 0.0

        self._stream = sd.InputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            blocksize=self.blocksize, device=self.device,
            callback=self._callback)
        self._stream.start()
        self._t_start = time.perf_counter()

    def stop(self):
        """Close the stream and return everything captured, aligned.

        Returns a dict with the mono audio, the sample rate, the key events
        (each carrying `t` in seconds from session start and `est_sample`,
        its estimated position in the audio) and capture health counters.
        """
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

        with self._lock:
            blocks = self._blocks
            anchors = list(self._anchors)
            events = list(self._events)
            overflows = self._overflows
            peak = self._peak
            self._blocks = []

        audio = (np.concatenate(blocks) if blocks
                 else np.zeros(0, dtype=np.float32))
        events = self._align(events, anchors, len(audio))
        return {
            "audio": audio,
            "sample_rate": self.sr,
            "events": events,
            "duration": len(audio) / float(self.sr) if self.sr else 0.0,
            "overflows": overflows,
            "peak": float(peak),
        }

    # -- input --------------------------------------------------------------

    def mark_key(self, keysym, char=""):
        """Record a keypress. Called from the Tk thread while recording."""
        t = time.perf_counter()
        label = keysym_to_label(keysym)
        with self._lock:
            if self._stream is None:
                return None
            event = {
                "t_perf": t,
                "keysym": keysym,
                "char": char,
                "label": label,
            }
            self._events.append(event)
            return event

    def _callback(self, indata, frames, time_info, status):
        now = time.perf_counter()
        if status:
            # An input overflow means samples were dropped, which shifts
            # everything after it - worth surfacing rather than swallowing.
            self._overflows += 1

        # PortAudio's inputBufferAdcTime is on its own stream clock; shifting
        # it by (perf_counter now - stream clock now) puts it on our clock
        # while keeping the device's input latency baked in.
        adc = getattr(time_info, "inputBufferAdcTime", 0.0) or 0.0
        current = getattr(time_info, "currentTime", 0.0) or 0.0
        if adc and current:
            t0 = adc + (now - current)
        else:
            t0 = now - frames / float(self.sr)

        block = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        level = float(np.sqrt(np.mean(np.square(block)))) if len(block) else 0.0
        block_peak = float(np.max(np.abs(block))) if len(block) else 0.0

        with self._lock:
            self._anchors.append((self._frames, t0))
            self._blocks.append(block)
            self._frames += len(block)
            self._level = level
            self._peak = max(self._peak, block_peak)

    # -- alignment ----------------------------------------------------------

    def _align(self, events, anchors, n_samples):
        """Attach `t` (seconds from start) and `est_sample` to each event."""
        if not events:
            return []
        anchor_times = [t0 for _, t0 in anchors]
        t_zero = anchor_times[0] if anchor_times else self._t_start
        aligned = []
        for i, event in enumerate(events):
            t = event["t_perf"]
            if anchor_times:
                # The anchor at or before this keypress dates the block the
                # click landed in, so drift over a long session cannot
                # accumulate the way a single session-start offset would.
                idx = max(0, bisect.bisect_right(anchor_times, t) - 1)
                frame, t0 = anchors[idx]
                est = int(round(frame + (t - t0) * self.sr))
            else:
                est = 0
            aligned.append({
                "index": i,
                "keysym": event["keysym"],
                "char": event["char"],
                "label": event["label"],
                "t": max(0.0, t - t_zero) if t_zero else 0.0,
                "est_sample": int(min(max(est, 0), max(n_samples - 1, 0))),
            })
        return aligned

    # -- live readouts ------------------------------------------------------

    @property
    def level(self):
        """Most recent block RMS, for the level meter."""
        with self._lock:
            return self._level

    @property
    def elapsed(self):
        return 0.0 if self._t_start is None else time.perf_counter() - self._t_start

    @property
    def key_count(self):
        with self._lock:
            return len(self._events)


# --------------------------------------------------------------------------
# Cutting a session into labelled clips
# --------------------------------------------------------------------------

class SegmentParams:
    """Clip window and quality thresholds.

    pre_ms/post_ms      clip window around the refined onset
    search_back/fwd_ms  where the true onset is hunted for, relative to the
                        key event's estimated position
    min_gap_ms          two keys closer than this share one sound: overlap
    repeat_gap_ms       same key closer than this is keyboard auto-repeat
    quiet_mult          a click must beat noise_floor * this to count
    """

    def __init__(self, pre_ms=20.0, post_ms=180.0, search_back_ms=45.0,
                 search_fwd_ms=70.0, min_gap_ms=80.0, repeat_gap_ms=55.0,
                 quiet_mult=3.0, keep_flagged=False):
        self.pre_ms = float(pre_ms)
        self.post_ms = float(post_ms)
        self.search_back_ms = float(search_back_ms)
        self.search_fwd_ms = float(search_fwd_ms)
        self.min_gap_ms = float(min_gap_ms)
        self.repeat_gap_ms = float(repeat_gap_ms)
        self.quiet_mult = float(quiet_mult)
        self.keep_flagged = bool(keep_flagged)

    def to_dict(self):
        return {
            "pre_ms": self.pre_ms, "post_ms": self.post_ms,
            "search_back_ms": self.search_back_ms,
            "search_fwd_ms": self.search_fwd_ms,
            "min_gap_ms": self.min_gap_ms,
            "repeat_gap_ms": self.repeat_gap_ms,
            "quiet_mult": self.quiet_mult,
        }

    @classmethod
    def from_args(cls, args):
        return cls(pre_ms=args.pre_ms, post_ms=args.post_ms,
                   min_gap_ms=args.min_gap_ms,
                   keep_flagged=args.keep_flagged)


def _frame_rms(y, win=512, hop=256):
    """RMS per frame, without pulling in librosa."""
    if len(y) < win:
        return np.array([float(np.sqrt(np.mean(np.square(y))))
                         if len(y) else 0.0])
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = y[idx]
    return np.sqrt(np.mean(np.square(frames), axis=1))


def noise_floor(y, sr=DEFAULT_SR):
    """A robust quiet-level estimate: the 20th percentile of frame RMS.

    Typing is mostly silence between clicks, so a low percentile lands in
    the room noise rather than on a keystroke.
    """
    if len(y) == 0:
        return 1e-6
    rms = _frame_rms(y)
    return max(float(np.percentile(rms, 20)), 1e-6)


def refine_onset(y, sr, est_sample, params):
    """Find the real click near `est_sample`.

    Returns (onset_sample, envelope_peak, raw_peak). The loudest part of the
    search window is the body of the click; the onset is where its attack
    begins, so the window is walked backwards until the signal drops under a
    fraction of that peak. Cropping from the attack rather than the peak
    keeps every clip aligned the same way.

    The walk-back runs on a 1ms-smoothed envelope rather than on raw
    samples: a click is a noise burst, and individual samples dip below any
    threshold mid-attack, which would strand the onset early. The raw peak
    comes back separately because that, not the envelope, is what says
    whether the input clipped.
    """
    back = int(sr * params.search_back_ms / 1000.0)
    fwd = int(sr * params.search_fwd_ms / 1000.0)
    lo = max(0, est_sample - back)
    hi = min(len(y), est_sample + fwd)
    if hi - lo < 8:
        return int(min(max(est_sample, 0), max(len(y) - 1, 0))), 0.0, 0.0

    window = np.abs(y[lo:hi])
    raw_peak = float(np.max(window))
    smooth = max(1, int(sr * 0.001))
    if smooth > 1 and len(window) > smooth * 2:
        env = np.convolve(window, np.ones(smooth) / smooth, mode="same")
    else:
        env = window

    peak_rel = int(np.argmax(env))
    peak_val = float(env[peak_rel])
    if peak_val <= 0.0:
        return lo + peak_rel, 0.0, raw_peak

    threshold = 0.15 * peak_val
    limit = max(0, peak_rel - int(sr * 0.02))   # attacks are short: <= 20ms
    i = peak_rel
    while i > limit and env[i] > threshold:
        i -= 1
    return lo + i, peak_val, raw_peak


def segment_session(audio, sr, events, params=None, floor=None):
    """Judge and place every key event; return one record per event.

    Each record carries where the clip is (`onset_sample`, `start`, `end`),
    how loud the click was, and a list of `flags`. An empty flag list means
    the clip is fit to train on.
    """
    params = params or SegmentParams()
    floor = noise_floor(audio, sr) if floor is None else floor
    pre = int(sr * params.pre_ms / 1000.0)
    post = int(sr * params.post_ms / 1000.0)

    records = []
    for i, event in enumerate(events):
        label = event.get("label")
        est = int(event.get("est_sample", 0))
        rec = {
            "index": event.get("index", i),
            "keysym": event.get("keysym", ""),
            "char": event.get("char", ""),
            "label": label,
            "t": float(event.get("t", 0.0)),
            "est_sample": est,
            "flags": [],
        }

        if not label:
            rec["flags"].append("unmapped")
            rec["onset_sample"] = est
            rec["start"] = est
            rec["end"] = est
            rec["peak"] = 0.0
            rec["raw_peak"] = 0.0
            records.append(rec)
            continue

        onset, peak, raw_peak = refine_onset(audio, sr, est, params)
        rec["onset_sample"] = int(onset)
        rec["peak"] = float(peak)
        rec["raw_peak"] = float(raw_peak)
        rec["start"] = max(0, int(onset) - pre)
        rec["end"] = min(len(audio), int(onset) + post)

        # The envelope peak is on the same scale as the frame-RMS noise
        # floor, so it is what decides "was there a click here at all".
        if peak < floor * params.quiet_mult:
            rec["flags"].append("quiet")
        if raw_peak >= 0.995:
            rec["flags"].append("clipped")
        records.append(rec)

    # Neighbour checks run over the event sequence, so they see keys that
    # were themselves rejected for another reason - a quiet click still
    # masks its neighbour's clip.
    gap = params.min_gap_ms / 1000.0
    repeat_gap = params.repeat_gap_ms / 1000.0
    for i, rec in enumerate(records):
        if not rec["label"]:
            continue
        prev = records[i - 1] if i > 0 else None
        nxt = records[i + 1] if i + 1 < len(records) else None
        if prev is not None and prev["label"]:
            delta = rec["t"] - prev["t"]
            if delta < repeat_gap and prev["label"] == rec["label"]:
                rec["flags"].append("repeat")
            elif delta < gap:
                rec["flags"].append("overlap")
        if nxt is not None and nxt["label"] and (nxt["t"] - rec["t"]) < gap:
            if "overlap" not in rec["flags"] and "repeat" not in rec["flags"]:
                rec["flags"].append("overlap")
    return records


def clip_for(audio, record, sr, params):
    """The fixed-length clip for a record, zero-padded if it ran off the end."""
    target = int(sr * params.pre_ms / 1000.0) + int(sr * params.post_ms / 1000.0)
    seg = audio[record["start"]:record["end"]]
    if len(seg) < target:
        seg = np.pad(seg, (0, target - len(seg)))
    return seg[:target]


def new_session_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def export_clips(data_dir, session_id, audio, sr, records, params):
    """Write one .wav per good record into data_dir/<label>/.

    Flagged records go to _rejected/<flag>/ when params.keep_flagged is set,
    and are simply counted otherwise. Returns a summary dict.
    """
    written = 0
    by_label = {}
    rejected = {}

    for rec in records:
        flags = rec["flags"]
        label = rec["label"]
        if flags or not label:
            reason = flags[0] if flags else "unmapped"
            rejected[reason] = rejected.get(reason, 0) + 1
            if params.keep_flagged and label:
                path = os.path.join(
                    data_dir, REJECTED_DIRNAME, reason,
                    f"{label}__{session_id}__{rec['index']:04d}.wav")
                write_wav(path, clip_for(audio, rec, sr, params), sr)
            continue

        path = os.path.join(
            data_dir, label, f"{label}__{session_id}__{rec['index']:04d}.wav")
        write_wav(path, clip_for(audio, rec, sr, params), sr)
        rec["clip"] = os.path.relpath(path, data_dir).replace("\\", "/")
        written += 1
        by_label[label] = by_label.get(label, 0) + 1

    return {"written": written, "by_label": by_label, "rejected": rejected}


def save_session(data_dir, session_id, audio, sr, records, params, extra=None):
    """Persist the raw recording plus a manifest of everything decided.

    The manifest is what makes --resegment possible later: it holds each
    event's estimated sample position, so no clock information is lost.
    """
    sessions_dir = os.path.join(data_dir, SESSIONS_DIRNAME)
    wav_path = os.path.join(sessions_dir, f"{session_id}.wav")
    json_path = os.path.join(sessions_dir, f"{session_id}.json")
    write_wav(wav_path, audio, sr)

    manifest = {
        "version": MANIFEST_VERSION,
        "session_id": session_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "sample_rate": int(sr),
        "duration": len(audio) / float(sr) if sr else 0.0,
        "wav": f"{session_id}.wav",
        "params": params.to_dict(),
        "events": [
            {k: rec[k] for k in ("index", "keysym", "char", "label", "t",
                                 "est_sample", "onset_sample", "peak",
                                 "raw_peak", "flags")
             if k in rec}
            for rec in records
        ],
    }
    if extra:
        manifest.update(extra)

    os.makedirs(sessions_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return wav_path, json_path


def record_session_to_dataset(data_dir, capture, params=None,
                              session_id=None, extra=None):
    """Slice a finished capture into the dataset and save the session.

    `capture` is what SessionRecorder.stop() returns. Returns a summary dict
    ready to print or show in the UI.
    """
    params = params or SegmentParams()
    session_id = session_id or new_session_id()
    audio = capture["audio"]
    sr = capture["sample_rate"]
    records = segment_session(audio, sr, capture["events"], params)

    summary = export_clips(data_dir, session_id, audio, sr, records, params)
    wav_path, json_path = save_session(
        data_dir, session_id, audio, sr, records, params, extra=extra)

    summary.update({
        "session_id": session_id,
        "events": len(records),
        "duration": capture.get("duration", 0.0),
        "overflows": capture.get("overflows", 0),
        "peak": capture.get("peak", 0.0),
        "session_wav": wav_path,
        "session_json": json_path,
    })
    return summary


def resegment(data_dir, params=None, session_ids=None, verbose=True):
    """Re-cut saved sessions with the current parameters.

    Existing clips from a session are removed first, so re-running is
    idempotent rather than additive.
    """
    params = params or SegmentParams()
    sessions_dir = os.path.join(data_dir, SESSIONS_DIRNAME)
    manifests = sorted(glob.glob(os.path.join(sessions_dir, "*.json")))
    if not manifests:
        raise ValueError(f"No saved sessions in {sessions_dir}")

    totals = {"sessions": 0, "written": 0, "rejected": {}}
    for path in manifests:
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        session_id = manifest.get(
            "session_id", os.path.splitext(os.path.basename(path))[0])
        if session_ids and session_id not in session_ids:
            continue

        wav_path = os.path.join(sessions_dir, manifest.get(
            "wav", f"{session_id}.wav"))
        if not os.path.exists(wav_path):
            if verbose:
                print(f"  [skip] {session_id}: recording missing")
            continue

        audio, sr = read_wav(wav_path)
        for stale in glob.glob(os.path.join(
                data_dir, "*", f"*__{session_id}__*.wav")):
            os.remove(stale)
        for stale in glob.glob(os.path.join(
                data_dir, REJECTED_DIRNAME, "*", f"*__{session_id}__*.wav")):
            os.remove(stale)

        records = segment_session(audio, sr, manifest.get("events", []), params)
        summary = export_clips(data_dir, session_id, audio, sr, records, params)
        save_session(data_dir, session_id, audio, sr, records, params)

        totals["sessions"] += 1
        totals["written"] += summary["written"]
        for reason, count in summary["rejected"].items():
            totals["rejected"][reason] = totals["rejected"].get(reason, 0) + count
        if verbose:
            print(f"  {session_id}: {summary['written']} clips"
                  f"{_rejected_note(summary['rejected'])}")

    if verbose:
        print(f"\nRe-cut {totals['sessions']} session(s): "
              f"{totals['written']} clips"
              f"{_rejected_note(totals['rejected'])}")
    return totals


def _rejected_note(rejected):
    if not rejected:
        return ""
    parts = ", ".join(f"{count} {reason}"
                      for reason, count in sorted(rejected.items()))
    return f"  (dropped: {parts})"


def dataset_counts(data_dir):
    """Clips currently on disk per label."""
    counts = {}
    if not os.path.isdir(data_dir):
        return counts
    for name in os.listdir(data_dir):
        if name.startswith("_"):
            continue
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            continue
        counts[name] = len(glob.glob(os.path.join(folder, "*.wav")))
    return counts


def weakest_labels(counts, n=8, pool=None):
    """The least-covered tracked labels, thinnest first."""
    pool = pool or TRACKED_LABELS
    ranked = sorted(pool, key=lambda label: (counts.get(label, 0),
                                             pool.index(label)))
    return ranked[:n]


def practice_text(labels, groups=9, group_size=4):
    """Random practice text built from the given labels.

    Real keystrokes need separating, so groups are joined with spaces (which
    are a tracked key themselves and always need samples anyway).
    """
    typeable = {
        "space": " ", "minus": "-", "equal": "=", "bracketleft": "[",
        "bracketright": "]", "backslash": "\\", "semicolon": ";",
        "apostrophe": "'", "grave": "`", "comma": ",", "period": ".",
        "slash": "/",
    }
    chars = [typeable.get(label, label) for label in labels
             if label in typeable or len(label) == 1]
    chars = [c for c in chars if c.strip()]
    if not chars:
        return random.choice(PROMPTS)
    return " ".join(
        "".join(random.choice(chars) for _ in range(group_size))
        for _ in range(groups))


# --------------------------------------------------------------------------
# Annotation UI
# --------------------------------------------------------------------------

class LevelMeter(tk.Canvas):
    """Rolling input level, drawn as a strip of bars.

    Bars are the cheapest readout that still shows a keystroke as a spike
    rather than a number twitching, which is what tells you the mic is
    actually picking the clicks up.
    """

    def __init__(self, parent, width=250, height=34, bars=50, bg=None):
        bg = bg or BG_PANEL
        super().__init__(parent, width=width, height=height, bg=bg, bd=0,
                         highlightthickness=0)
        self.bar_count = bars
        self.values = deque([0.0] * bars, maxlen=bars)
        self.bind("<Configure>", lambda e: self._redraw())

    def push(self, level):
        # A log-ish scale keeps quiet room noise visible while leaving room
        # at the top for a click, which is 20-40dB above it.
        scaled = 0.0 if level <= 0 else min(1.0, (level ** 0.45) * 1.8)
        self.values.append(scaled)
        self._redraw()

    def clear(self):
        self.values = deque([0.0] * self.bar_count, maxlen=self.bar_count)
        self._redraw()

    def _redraw(self):
        self.delete("all")
        width = max(self.winfo_width(), int(self["width"]))
        height = max(self.winfo_height(), int(self["height"]))
        mid = height / 2
        slot = width / float(self.bar_count)
        for i, value in enumerate(self.values):
            x = i * slot
            half = max(1.0, value * (height - 6) / 2)
            colour = (WARN if value >= 0.97
                      else blend(FG_DIM, ACCENT, min(1.0, value * 1.6)))
            self.create_rectangle(x + 1, mid - half, x + slot - 1, mid + half,
                                  fill=colour, outline="")


class CoverageGrid(tk.Canvas):
    """One chip per tracked key, shaded by how many clips it has.

    This is the panel that decides what to type next: it makes a thin key
    obvious at a glance instead of requiring a folder count.
    """

    CHIP_W = 46
    CHIP_H = 26
    GAP = 5

    def __init__(self, parent, columns=13, target=DEFAULT_TARGET, bg=None):
        self.columns = columns
        self.target = target
        self.counts = {}
        self.pending = {}
        bg = bg or BG_PANEL
        rows = -(-len(TRACKED_LABELS) // columns)
        super().__init__(
            parent, bd=0, highlightthickness=0, bg=bg,
            width=columns * (self.CHIP_W + self.GAP),
            height=rows * (self.CHIP_H + self.GAP))
        self.surface = bg
        self._redraw()

    def set_counts(self, counts, pending=None):
        self.counts = dict(counts or {})
        self.pending = dict(pending or {})
        self._redraw()

    def _redraw(self):
        self.delete("all")
        for i, label in enumerate(TRACKED_LABELS):
            col, row = i % self.columns, i // self.columns
            x = col * (self.CHIP_W + self.GAP)
            y = row * (self.CHIP_H + self.GAP)
            done = self.counts.get(label, 0)
            waiting = self.pending.get(label, 0)
            total = done + waiting
            fullness = min(1.0, total / float(self.target)) if self.target else 0.0

            fill = blend(self.surface, OK, 0.10 + 0.45 * fullness) if total \
                else blend(self.surface, FG_DIM, 0.12)
            # A chip holding un-saved keystrokes is outlined in the accent,
            # so the current session's contribution reads separately from
            # what is already banked on disk.
            outline = ACCENT if waiting else blend(self.surface, FG_DIM, 0.35)
            self.create_polygon(
                rounded_rect_points(x, y, x + self.CHIP_W, y + self.CHIP_H, 7),
                smooth=True, splinesteps=16, fill=fill, outline=outline)
            self.create_text(x + 8, y + self.CHIP_H / 2, anchor="w",
                             text=label_display(label), font=mono(9),
                             fill=FG_TEXT if total else FG_DIM)
            self.create_text(x + self.CHIP_W - 7, y + self.CHIP_H / 2,
                             anchor="e", text=str(total), font=mono(9),
                             fill=FG_LABEL if total else FG_DIM)


class AnnotatorApp:
    """Hit record, type, stop. The dataset grows by itself.

    Keystrokes are bound at the toplevel rather than in a focused text
    widget: there is no click-to-focus step, and the app can label a key and
    echo it without a real entry field swallowing anything.
    """

    def __init__(self, root, data_dir=DEFAULT_DATA_DIR, params=None,
                 target=DEFAULT_TARGET, sr=DEFAULT_SR, device=None):
        self.root = root
        init_fonts(root)

        root.title("keyloggd - keystroke annotation")
        root.configure(bg=BG)
        root.geometry("1010x800")
        root.minsize(900, 720)

        self.data_dir = os.path.abspath(data_dir)
        self.params = params or SegmentParams()
        self.target = target
        self.sr = sr
        self.device = device

        self.recorder = None
        self.recording = False
        self.pending = {}          # labels typed this session, not yet saved
        self.typed_chars = 0
        self.results = queue.Queue()
        self.prompt_index = 0
        self._coverage_dirty = False

        self._build_layout()
        self._refresh_coverage()
        self.root.bind("<KeyPress>", self._on_keypress)
        self.root.bind("<F9>", lambda e: self._toggle_record())
        self.root.bind("<Control-r>", lambda e: self._toggle_record())
        self.root.bind("<Escape>", lambda e: self._stop_recording())

        if not SessionRecorder.available():
            self.record_btn.set_enabled(False)
            self._status("recording needs the 'sounddevice' package - "
                         "pip install sounddevice", WARN)
        self.root.after(60, self._tick)
        self.root.after(80, self._drain_results)

    # -- layout -------------------------------------------------------------

    def _build_layout(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=26, pady=22)

        self._build_header(outer)
        self._build_recorder(outer)
        self._build_typing(outer)
        self._build_coverage(outer)
        self._build_footer(outer)

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=BG)
        header.pack(fill="x", pady=(0, 16))

        left = tk.Frame(header, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="keystroke annotation", bg=BG, fg=FG_TEXT,
                 font=ui(22)).pack(anchor="w")
        tk.Label(left, bg=BG, fg=FG_LABEL, font=ui(11),
                 text="hit record and type - every key labels its own sound"
                 ).pack(anchor="w", pady=(2, 0))

        right = tk.Frame(header, bg=BG)
        right.pack(side="right")
        pill = RoundedPanel(right, BG_INPUT, radius=8, padding=5, bg=BG)
        pill.pack(side="left", padx=(0, 8))
        self.dir_label = tk.Label(pill.inner, bg=BG_INPUT, fg=FG_LABEL,
                                  font=mono(10), padx=8, pady=3,
                                  text=self._short_dir())
        self.dir_label.pack(side="left")
        PillButton(right, "change folder", self._choose_data_dir, size=10,
                   bg=BG, padx=12, pady=6).pack(side="left")

    def _build_recorder(self, parent):
        panel = RoundedPanel(parent, BG_PANEL, radius=14, padding=16, bg=BG)
        panel.pack(fill="x")
        inner = panel.inner

        row = tk.Frame(inner, bg=BG_PANEL)
        row.pack(fill="x")

        self.record_btn = PillButton(row, "start recording", self._toggle_record,
                                     kind="primary", size=12, bg=BG_PANEL,
                                     padx=22, pady=11)
        self.record_btn.pack(side="left")

        self.dot = tk.Canvas(row, width=14, height=14, bg=BG_PANEL, bd=0,
                             highlightthickness=0)
        self.dot.pack(side="left", padx=(14, 8))
        self._dot_item = self.dot.create_oval(2, 2, 12, 12, fill=FG_DIM,
                                              outline="")

        self.timer_label = tk.Label(row, text="00:00.0", bg=BG_PANEL,
                                    fg=FG_TEXT, font=mono(18))
        self.timer_label.pack(side="left")

        self.meter = LevelMeter(row, bg=BG_PANEL)
        self.meter.pack(side="left", padx=18)

        counts = tk.Frame(row, bg=BG_PANEL)
        counts.pack(side="right")
        self.keys_label = tk.Label(counts, text="0", bg=BG_PANEL, fg=FG_TEXT,
                                   font=mono(18))
        self.keys_label.pack(side="right")
        tk.Label(counts, text="keys captured  ", bg=BG_PANEL, fg=FG_LABEL,
                 font=ui(11)).pack(side="right")

        hint = tk.Label(
            inner, bg=BG_PANEL, fg=FG_DIM, font=ui(10), justify="left",
            text="ctrl+r or F9 toggles recording   |   esc stops and saves   "
                 "|   escape and the modifiers are never labelled")
        hint.pack(anchor="w", pady=(12, 0))

    def _build_typing(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True, pady=(16, 0))

        prompt_row = tk.Frame(wrap, bg=BG)
        prompt_row.pack(fill="x", pady=(0, 8))
        tk.Label(prompt_row, text="prompt", bg=BG, fg=FG_LABEL,
                 font=ui(11)).pack(side="left", padx=(0, 10))
        self.prompt_label = tk.Label(prompt_row, bg=BG, fg=FG_LABEL,
                                     font=mono(11), anchor="w",
                                     text=PROMPTS[0])
        self.prompt_label.pack(side="left", fill="x", expand=True)
        PillButton(prompt_row, "weak keys", self._prompt_weak_keys, size=10,
                   bg=BG, padx=12, pady=6).pack(side="right")
        PillButton(prompt_row, "next prompt", self._next_prompt, size=10,
                   bg=BG, padx=12, pady=6).pack(side="right", padx=(0, 8))

        panel = RoundedPanel(wrap, BG_INPUT, radius=12, padding=10, bg=BG)
        panel.pack(fill="both", expand=True)
        self.typed = tk.Text(
            panel.inner, height=7, width=88, bg=BG_INPUT, fg=FG_TEXT,
            font=mono(13), wrap="char", relief="flat", bd=0,
            highlightthickness=0, padx=8, pady=6, insertwidth=0,
            state="disabled", selectbackground=blend(BG_INPUT, ACCENT, 0.35))
        self.typed.pack(fill="both", expand=True)
        # Flagged keys are tinted where they were typed, so a session's
        # problems are visible in the text rather than only in the summary.
        self.typed.tag_configure("flagged", foreground=FG_INCORRECT)
        self.typed.tag_configure("idle", foreground=FG_DIM)
        self._set_typed_placeholder()

    def _build_coverage(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="x", pady=(16, 0))

        head = tk.Frame(wrap, bg=BG)
        head.pack(fill="x", pady=(0, 8))
        tk.Label(head, text="coverage", bg=BG, fg=FG_TEXT,
                 font=ui(13)).pack(side="left")
        self.coverage_note = tk.Label(head, bg=BG, fg=FG_LABEL, font=ui(10),
                                      text=f"target {self.target} clips per key")
        self.coverage_note.pack(side="left", padx=(10, 0))

        panel = RoundedPanel(wrap, BG_PANEL, radius=12, padding=12, bg=BG)
        panel.pack(anchor="w")
        self.coverage = CoverageGrid(panel.inner, target=self.target,
                                     bg=BG_PANEL)
        self.coverage.pack()

    def _build_footer(self, parent):
        footer = tk.Frame(parent, bg=BG)
        footer.pack(fill="x", pady=(16, 0))
        # The spinner packs and unpacks itself, so it gets its own holder on
        # the left to keep it ahead of the status text when it appears.
        holder = tk.Frame(footer, bg=BG)
        holder.pack(side="left")
        self.spinner = Spinner(holder, bg=BG)
        self.status_label = tk.Label(footer, bg=BG, fg=FG_LABEL, font=ui(11),
                                     anchor="w", justify="left",
                                     text="ready - hit record and start typing")
        self.status_label.pack(side="left", padx=(8, 0))

    # -- small helpers ------------------------------------------------------

    def _short_dir(self):
        path = self.data_dir
        return path if len(path) <= 46 else "..." + path[-43:]

    def _status(self, text, colour=None):
        self.status_label.configure(text=text, fg=colour or FG_LABEL)

    def _set_typed_placeholder(self):
        self.typed.configure(state="normal")
        self.typed.delete("1.0", "end")
        self.typed.insert("end", "your keystrokes appear here while recording",
                          "idle")
        self.typed.configure(state="disabled")

    # -- recording ----------------------------------------------------------

    def _toggle_record(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.recording:
            return
        self.recorder = SessionRecorder(sr=self.sr, device=self.device)
        try:
            self.recorder.start()
        except Exception as exc:
            self.recorder = None
            self._status(f"could not start recording: {exc}", FG_INCORRECT)
            return

        self.recording = True
        self.pending = {}
        self.typed_chars = 0
        self.record_btn.set_text("stop and save")
        self.dot.itemconfigure(self._dot_item, fill=ACCENT)
        self.typed.configure(state="normal")
        self.typed.delete("1.0", "end")
        self.typed.configure(state="disabled")
        self.meter.clear()
        self._status("recording - type away, esc or ctrl+r to stop", ACCENT)

    def _stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        recorder, self.recorder = self.recorder, None
        self.record_btn.set_text("start recording")
        self.record_btn.set_enabled(False)
        self.dot.itemconfigure(self._dot_item, fill=FG_DIM)
        self.meter.clear()

        capture = recorder.stop()
        if not capture["events"]:
            self.record_btn.set_enabled(True)
            self._status("stopped - no keystrokes captured, nothing saved",
                         WARN)
            self._set_typed_placeholder()
            return

        self._status(f"cutting {len(capture['events'])} keystrokes...", FG_LABEL)
        self.spinner.start(side="left")
        # Segmenting a few thousand clips and writing them out is disk-bound;
        # doing it on the Tk thread would freeze the window mid-save.
        threading.Thread(target=self._segment_worker, args=(capture,),
                         daemon=True).start()

    def _segment_worker(self, capture):
        try:
            summary = record_session_to_dataset(
                self.data_dir, capture, self.params,
                extra={"tool": "annotator"})
            self.results.put(("ok", summary))
        except Exception as exc:                            # pragma: no cover
            self.results.put(("error", exc))

    def _drain_results(self):
        while True:
            try:
                kind, payload = self.results.get_nowait()
            except queue.Empty:
                break
            self.spinner.stop()
            self.record_btn.set_enabled(SessionRecorder.available())
            if kind == "error":
                self._status(f"saving failed: {payload}", FG_INCORRECT)
                continue
            self.pending = {}
            self._refresh_coverage()
            self._show_summary(payload)
        self.root.after(80, self._drain_results)

    def _show_summary(self, summary):
        parts = [f"saved {summary['written']} clips from "
                 f"{summary['events']} keystrokes",
                 f"{summary['duration']:.1f}s"]
        if summary["rejected"]:
            parts.append("dropped " + ", ".join(
                f"{count} {reason}"
                for reason, count in sorted(summary["rejected"].items())))
        if summary["overflows"]:
            parts.append(f"{summary['overflows']} input overflow(s) - "
                         "alignment may have drifted")
        parts.append(f"session {summary['session_id']}")
        colour = WARN if (summary["overflows"] or not summary["written"]) else OK
        self._status("  |  ".join(parts), colour)

    # -- keystrokes ---------------------------------------------------------

    def _on_keypress(self, event):
        if not self.recording or self.recorder is None:
            return
        # Control-chords are shortcuts, not typing - and their key sound is
        # buried under a held modifier anyway.
        if event.state & 0x0004:
            return
        if event.keysym in IGNORED_KEYSYMS:
            return

        marked = self.recorder.mark_key(event.keysym, event.char or "")
        if marked is None:
            return
        label = marked["label"]
        if label:
            self.pending[label] = self.pending.get(label, 0) + 1
            # Redrawing 60 chips per keystroke would fight fast typing, so
            # the grid catches up on the next tick instead.
            self._coverage_dirty = True
        self._echo(event, labelled=bool(label))

    def _echo(self, event, labelled):
        """Mirror the keystroke into the transcript panel."""
        keysym = event.keysym
        if keysym == "BackSpace":
            # Backspace is a real key with a real sound, so it is recorded -
            # but it should still look like a backspace in the transcript.
            self.typed.configure(state="normal")
            self.typed.delete("end-2c", "end-1c")
            self.typed.configure(state="disabled")
            self.typed_chars = max(0, self.typed_chars - 1)
            return

        text = {"Return": "\n", "Tab": "    ", "space": " "}.get(
            keysym, event.char)
        if not text:
            return
        self.typed.configure(state="normal")
        self.typed.insert("end", text, () if labelled else ("flagged",))
        self.typed.see("end")
        self.typed.configure(state="disabled")
        self.typed_chars += 1

    # -- periodic updates ---------------------------------------------------

    def _tick(self):
        if self.recording and self.recorder is not None:
            elapsed = self.recorder.elapsed
            minutes, seconds = divmod(elapsed, 60)
            self.timer_label.configure(
                text=f"{int(minutes):02d}:{seconds:04.1f}")
            self.keys_label.configure(text=str(self.recorder.key_count))
            self.meter.push(self.recorder.level)
        if self._coverage_dirty:
            self._coverage_dirty = False
            self.coverage.set_counts(self.coverage.counts, self.pending)
        self.root.after(60, self._tick)

    def _refresh_coverage(self):
        counts = dataset_counts(self.data_dir)
        self.coverage.set_counts(counts, self.pending)
        total = sum(counts.values())
        thin = sum(1 for label in TRACKED_LABELS
                   if counts.get(label, 0) < self.target)
        self.coverage_note.configure(
            text=f"target {self.target} clips per key   |   {total} clips on "
                 f"disk   |   {thin} of {len(TRACKED_LABELS)} keys still thin")

    # -- actions ------------------------------------------------------------

    def _choose_data_dir(self):
        if self.recording:
            self._status("stop recording before changing folder", WARN)
            return
        path = filedialog.askdirectory(title="Dataset folder",
                                       initialdir=self.data_dir)
        if path:
            self.data_dir = os.path.abspath(path)
            self.dir_label.configure(text=self._short_dir())
            self._refresh_coverage()
            self._status(f"dataset folder: {self.data_dir}")

    def _next_prompt(self):
        self.prompt_index = (self.prompt_index + 1) % len(PROMPTS)
        self.prompt_label.configure(text=PROMPTS[self.prompt_index], fg=FG_LABEL)

    def _prompt_weak_keys(self):
        counts = dataset_counts(self.data_dir)
        weak = weakest_labels(counts, n=8)
        self.prompt_label.configure(text=practice_text(weak), fg=FG_TEXT)
        self._status("prompt rebuilt from the least-covered keys: "
                     + " ".join(label_display(l) for l in weak))


def launch_annotator(data_dir=DEFAULT_DATA_DIR, params=None,
                     target=DEFAULT_TARGET, sr=DEFAULT_SR, device=None):
    """Open the annotation platform."""
    if _THEME_ERROR is not None:
        sys.exit("The annotation UI needs this project's theme.py and anim.py "
                 f"alongside it ({_THEME_ERROR}).")
    os.makedirs(data_dir, exist_ok=True)
    root = tk.Tk()
    AnnotatorApp(root, data_dir=data_dir, params=params, target=target,
                 sr=sr, device=device)
    root.mainloop()


# --------------------------------------------------------------------------
# Audio utilities (training side)
# --------------------------------------------------------------------------

def load_audio(path, sr=DEFAULT_SR):
    import librosa
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def extract_keystroke_segment(y, sr, pre_ms=20, post_ms=180, top_db=30):
    """
    If a clip contains silence around the keystroke (or you fed in a longer
    continuous recording), isolate the loudest onset and crop a fixed
    window around it. This makes clip length/alignment consistent across
    examples, which matters a lot for feature quality.
    """
    import librosa
    # Find non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        onset_sample = int(np.argmax(np.abs(y)))
    else:
        # Take the loudest interval's peak as the onset
        peak_val = -1
        onset_sample = 0
        for start, end in intervals:
            seg = y[start:end]
            local_peak = np.max(np.abs(seg)) if len(seg) else 0
            if local_peak > peak_val:
                peak_val = local_peak
                onset_sample = start + int(np.argmax(np.abs(seg)))

    pre = int(sr * pre_ms / 1000)
    post = int(sr * post_ms / 1000)
    start = max(0, onset_sample - pre)
    end = min(len(y), onset_sample + post)
    segment = y[start:end]

    # Pad to fixed length so all feature vectors line up
    target_len = pre + post
    if len(segment) < target_len:
        segment = np.pad(segment, (0, target_len - len(segment)))
    else:
        segment = segment[:target_len]
    return segment


def extract_features(y, sr, n_mfcc=20):
    """
    Extract a fixed-length feature vector combining MFCCs (timbre),
    spectral centroid/bandwidth/rolloff (brightness/shape), zero-crossing
    rate (percussiveness), and RMS energy envelope stats - all useful for
    telling apart the short, percussive clicks of different keys.
    """
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    def stats(x):
        return np.concatenate([x.mean(axis=1), x.std(axis=1)])

    feature_vec = np.concatenate([
        stats(mfcc),
        stats(mfcc_delta),
        stats(spec_centroid),
        stats(spec_bandwidth),
        stats(spec_rolloff),
        stats(zcr),
        stats(rms),
    ])
    return feature_vec


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------

def build_dataset(data_dir, sr=DEFAULT_SR, auto_segment=True):
    # Underscore-prefixed folders are the annotator's bookkeeping
    # (_sessions, _rejected), not keys.
    labels = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("_")
    ])
    if not labels:
        raise ValueError(f"No label sub-folders found in {data_dir}")

    X, y_labels = [], []
    for label in labels:
        files = glob.glob(os.path.join(data_dir, label, "*.wav"))
        files += glob.glob(os.path.join(data_dir, label, "*.mp3"))
        if not files:
            print(f"  [warn] no audio files found for label '{label}'")
            continue
        print(f"  loading '{label}': {len(files)} files")
        for f in files:
            try:
                audio, sr_ = load_audio(f, sr=sr)
                if auto_segment:
                    audio = extract_keystroke_segment(audio, sr_)
                feats = extract_features(audio, sr_)
                X.append(feats)
                y_labels.append(label)
            except Exception as e:
                print(f"    [skip] {f}: {e}")

    if not X:
        raise ValueError("No usable audio was loaded. Check your dataset directory.")

    return np.array(X), np.array(y_labels)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(data_dir, model_out, sr=DEFAULT_SR):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    print(f"Building dataset from {data_dir} ...")
    X, y_labels = build_dataset(data_dir, sr=sr)
    print(f"Total examples: {len(X)}, classes: {sorted(set(y_labels))}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForestClassifier ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    bundle = {
        "model": clf,
        "scaler": scaler,
        "label_encoder": encoder,
        "sample_rate": sr,
    }
    joblib.dump(bundle, model_out)
    print(f"\nSaved model to {model_out}")


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def predict(clip_path, model_path):
    import joblib

    bundle = joblib.load(model_path)
    clf = bundle["model"]
    scaler = bundle["scaler"]
    encoder = bundle["label_encoder"]
    sr = bundle["sample_rate"]

    audio, sr_ = load_audio(clip_path, sr=sr)
    audio = extract_keystroke_segment(audio, sr_)
    feats = extract_features(audio, sr_).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    pred = clf.predict(feats_scaled)[0]
    proba = clf.predict_proba(feats_scaled)[0]
    label = encoder.inverse_transform([pred])[0]

    top5_idx = np.argsort(proba)[::-1][:5]
    print(f"Predicted key: {label}")
    print("Top candidates:")
    for i in top5_idx:
        print(f"  {encoder.classes_[i]:>8s}  {proba[i]*100:5.1f}%")

    return label


# --------------------------------------------------------------------------
# Inference platform - read typing back from sound alone
# --------------------------------------------------------------------------
#
# The annotator has the keyboard tell it where every click is. Inference gets
# no such help: it is handed only audio and has to find the keystrokes itself,
# classify each with the trained model, and reconstruct what was typed. When
# the operator also types the real text (scoring mode) the two are lined up so
# the attack's accuracy can be read off directly.

# Best-guess character each label stands for, for reconstructing readable
# text. Physical-key labels (e.g. "semicolon") map back to their character;
# backspace is handled by reconstruct_text rather than printed.
LABEL_TO_CHAR = {c: c for c in string.ascii_lowercase}
LABEL_TO_CHAR.update({d: d for d in "0123456789"})
LABEL_TO_CHAR.update({
    "space": " ", "enter": "\n", "tab": "\t",
    "minus": "-", "equal": "=", "bracketleft": "[", "bracketright": "]",
    "backslash": "\\", "semicolon": ";", "apostrophe": "'", "grave": "`",
    "comma": ",", "period": ".", "slash": "/",
})
LABEL_TO_CHAR.update({f"kp_{d}": d for d in "0123456789"})


def label_to_char(label):
    """Readable character for a predicted label, bracketed if it has none."""
    if label in LABEL_TO_CHAR:
        return LABEL_TO_CHAR[label]
    return label if len(label) == 1 else f"[{label}]"


def reconstruct_text(predictions):
    """Stitch a predicted key sequence back into text.

    A predicted backspace deletes the previous character, the way it would
    have while typing, so the reconstruction reads as the finished line
    rather than as a raw key log.
    """
    out = []
    for pred in predictions:
        label = pred["label"]
        if label == "backspace":
            if out:
                out.pop()
            continue
        out.append(label_to_char(label))
    return "".join(out)


def load_bundle(model_path):
    """Load a trained model bundle, with a clear message if deps are absent."""
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError(
            "inference needs joblib + scikit-learn:\n"
            "    pip install scikit-learn joblib") from exc
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"no model at {model_path} - train one first")
    return joblib.load(model_path)


def detect_onsets(audio, sr, floor=None, mult=4.0, min_gap_ms=70.0,
                  smooth_ms=1.0):
    """Find keystroke clicks in continuous audio (numpy only).

    A click is a sharp rise in the smoothed envelope above the room-noise
    floor. Each detection claims a refractory window of `min_gap_ms`, which
    both prevents a single click registering twice and sets the closest two
    keystrokes that can be told apart - the same limit the annotator uses to
    call overlaps.
    """
    n = len(audio)
    if n == 0:
        return []
    floor = noise_floor(audio, sr) if floor is None else floor
    threshold = floor * mult

    env = np.abs(audio)
    smooth = max(1, int(sr * smooth_ms / 1000.0))
    if smooth > 1 and n > smooth * 2:
        env = np.convolve(env, np.ones(smooth) / smooth, mode="same")

    gap = max(1, int(sr * min_gap_ms / 1000.0))
    onsets = []
    i = 0
    while i < n:
        if env[i] > threshold:
            hi = min(n, i + gap)
            peak = i + int(np.argmax(env[i:hi]))
            onsets.append(peak)
            i = peak + gap                 # refractory: skip this click's tail
        else:
            i += 1
    return onsets


def _crop_window(audio, onset, sr, params):
    """Fixed-length clip around an onset, matching the annotator's windows."""
    pre = int(sr * params.pre_ms / 1000.0)
    post = int(sr * params.post_ms / 1000.0)
    start = max(0, int(onset) - pre)
    end = min(len(audio), int(onset) + post)
    seg = audio[start:end]
    target = pre + post
    if len(seg) < target:
        seg = np.pad(seg, (0, target - len(seg)))
    return seg[:target]


def classify_clip(clip, sr, bundle, feature_fn=None):
    """Classify one clip; return (label, top5, confidence).

    `feature_fn` defaults to the same extract_features used in training, and
    is injectable so the pipeline can be exercised without librosa.
    """
    feature_fn = feature_fn or extract_features
    clf = bundle["model"]
    scaler = bundle.get("scaler")
    encoder = bundle["label_encoder"]

    feats = np.asarray(feature_fn(clip, sr), dtype=float).reshape(1, -1)
    if scaler is not None:
        feats = scaler.transform(feats)
    proba = np.asarray(clf.predict_proba(feats)[0], dtype=float)
    order = np.argsort(proba)[::-1]
    classes = encoder.classes_
    top = [(str(classes[j]), float(proba[j])) for j in order[:5]]
    return top[0][0], top, float(proba[order[0]])


def infer_audio(audio, sr, bundle, params=None, detect_mult=4.0,
                min_gap_ms=None, feature_fn=None):
    """Detect, crop and classify every keystroke in a recording.

    Returns one dict per detected click: its position (`onset_sample`, `t`
    seconds), the predicted `label`, its `confidence`, and the `top` five
    candidates.
    """
    params = params or SegmentParams()
    gap = params.min_gap_ms if min_gap_ms is None else min_gap_ms
    onsets = detect_onsets(audio, sr, mult=detect_mult, min_gap_ms=gap)

    preds = []
    for i, peak in enumerate(onsets):
        # Re-use the training-time walk-back so the clip handed to the model
        # is framed exactly as its training clips were.
        onset, _, _ = refine_onset(audio, sr, peak, params)
        clip = _crop_window(audio, onset, sr, params)
        label, top, conf = classify_clip(clip, sr, bundle, feature_fn)
        preds.append({
            "index": i,
            "onset_sample": int(onset),
            "t": int(onset) / float(sr) if sr else 0.0,
            "label": label,
            "confidence": conf,
            "top": top,
        })
    return preds


def score_predictions(predictions, events, sr, tolerance_ms=70.0):
    """Line predictions up against what was actually typed and score them.

    Each true keypress claims the nearest still-unclaimed prediction within
    `tolerance_ms`; a claimed one is correct when its label matches. Reports
    accuracy plus the ways it can go wrong: missed keys (no detection near a
    real press) and spurious ones (a detection near no real press).
    """
    tol = tolerance_ms / 1000.0
    truths = [e for e in events if e.get("label")]
    claimed = [False] * len(predictions)
    matched = []
    correct = 0
    for ev in truths:
        best, best_d = -1, tol + 1.0
        for j, pred in enumerate(predictions):
            if claimed[j]:
                continue
            d = abs(pred["t"] - ev["t"])
            if d < best_d:
                best, best_d = j, d
        if best >= 0 and best_d <= tol:
            claimed[best] = True
            ok = predictions[best]["label"] == ev["label"]
            correct += int(ok)
            matched.append({"true": ev["label"],
                            "pred": predictions[best]["label"],
                            "ok": ok,
                            "confidence": predictions[best]["confidence"]})
        else:
            matched.append({"true": ev["label"], "pred": None, "ok": False,
                            "confidence": 0.0})
    total = len(truths)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "detected": len(predictions),
        "missed": sum(1 for m in matched if m["pred"] is None),
        "spurious": sum(1 for c in claimed if not c),
        "matched": matched,
    }


def infer_file(clip_path, model_path, params=None):
    """Headless: reconstruct the typing in a recording and print it."""
    bundle = load_bundle(model_path)
    sr = bundle.get("sample_rate", DEFAULT_SR)
    audio, sr_ = load_audio(clip_path, sr=sr)
    preds = infer_audio(audio, sr_, bundle, params=params)
    text = reconstruct_text(preds)
    print(f"Detected {len(preds)} keystrokes in {clip_path}")
    print(f"Reconstructed: {text!r}\n")
    for pred in preds:
        alts = "  ".join(f"{name}:{p*100:.0f}%" for name, p in pred["top"][:3])
        print(f"  t={pred['t']:6.3f}s  {pred['label']:>8s}  "
              f"{pred['confidence']*100:5.1f}%   ({alts})")
    return preds


# --------------------------------------------------------------------------
# Inference UI
# --------------------------------------------------------------------------

class InferenceApp:
    """Load a trained model, record typing, read it back from the sound.

    Tick "score against what I type" and the real keystrokes are captured
    alongside the audio, so the finish screen can show how much of the typing
    the model recovered - a live read on how strong the acoustic leak is.
    """

    def __init__(self, root, model_path="keystroke_model.joblib",
                 params=None, sr=DEFAULT_SR, device=None):
        self.root = root
        init_fonts(root)

        root.title("keyloggd - keystroke inference")
        root.configure(bg=BG)
        root.geometry("1010x800")
        root.minsize(900, 720)

        self.model_path = os.path.abspath(model_path)
        self.params = params or SegmentParams()
        self.sr = sr
        self.device = device

        self.bundle = None
        self.recorder = None
        self.recording = False
        self.score_var = tk.BooleanVar(value=True)
        self.results = queue.Queue()

        self._build_layout()
        self.root.bind("<KeyPress>", self._on_keypress)
        self.root.bind("<F9>", lambda e: self._toggle_record())
        self.root.bind("<Control-r>", lambda e: self._toggle_record())
        self.root.bind("<Escape>", lambda e: self._stop_recording())

        self._try_load(self.model_path, announce=False)
        if not SessionRecorder.available():
            self._status("recording needs the 'sounddevice' package - "
                         "pip install sounddevice", WARN)
        self.root.after(60, self._tick)
        self.root.after(80, self._drain_results)

    # -- layout -------------------------------------------------------------

    def _build_layout(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=26, pady=22)
        self._build_header(outer)
        self._build_recorder(outer)
        self._build_reconstruction(outer)
        self._build_detail(outer)
        self._build_footer(outer)

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=BG)
        header.pack(fill="x", pady=(0, 16))
        left = tk.Frame(header, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="keystroke inference", bg=BG, fg=FG_TEXT,
                 font=ui(22)).pack(anchor="w")
        tk.Label(left, bg=BG, fg=FG_LABEL, font=ui(11),
                 text="record typing - the model reads it back from the sound"
                 ).pack(anchor="w", pady=(2, 0))

        right = tk.Frame(header, bg=BG)
        right.pack(side="right")
        pill = RoundedPanel(right, BG_INPUT, radius=8, padding=5, bg=BG)
        pill.pack(side="left", padx=(0, 8))
        self.model_label = tk.Label(pill.inner, bg=BG_INPUT, fg=FG_LABEL,
                                    font=mono(10), padx=8, pady=3,
                                    text="no model loaded")
        self.model_label.pack(side="left")
        PillButton(right, "load model", self._choose_model, size=10, bg=BG,
                   padx=12, pady=6).pack(side="left")

    def _build_recorder(self, parent):
        panel = RoundedPanel(parent, BG_PANEL, radius=14, padding=16, bg=BG)
        panel.pack(fill="x")
        inner = panel.inner

        row = tk.Frame(inner, bg=BG_PANEL)
        row.pack(fill="x")
        self.record_btn = PillButton(row, "start listening", self._toggle_record,
                                     kind="primary", size=12, bg=BG_PANEL,
                                     padx=22, pady=11)
        self.record_btn.pack(side="left")
        self.record_btn.set_enabled(False)

        self.dot = tk.Canvas(row, width=14, height=14, bg=BG_PANEL, bd=0,
                             highlightthickness=0)
        self.dot.pack(side="left", padx=(14, 8))
        self._dot_item = self.dot.create_oval(2, 2, 12, 12, fill=FG_DIM,
                                              outline="")
        self.timer_label = tk.Label(row, text="00:00.0", bg=BG_PANEL,
                                    fg=FG_TEXT, font=mono(18))
        self.timer_label.pack(side="left")
        self.meter = LevelMeter(row, bg=BG_PANEL)
        self.meter.pack(side="left", padx=18)

        counts = tk.Frame(row, bg=BG_PANEL)
        counts.pack(side="right")
        self.heard_label = tk.Label(counts, text="0", bg=BG_PANEL, fg=FG_TEXT,
                                    font=mono(18))
        self.heard_label.pack(side="right")
        tk.Label(counts, text="keys typed  ", bg=BG_PANEL, fg=FG_LABEL,
                 font=ui(11)).pack(side="right")

        toggle_row = tk.Frame(inner, bg=BG_PANEL)
        toggle_row.pack(fill="x", pady=(12, 0))
        self.score_toggle = tk.Label(
            toggle_row, bg=BG_PANEL, fg=FG_TEXT, font=ui(11), cursor="hand2")
        self.score_toggle.pack(side="left")
        self.score_toggle.bind("<Button-1>", lambda e: self._toggle_score())
        self._render_score_toggle()
        tk.Label(toggle_row, bg=BG_PANEL, fg=FG_DIM, font=ui(10),
                 text="   ctrl+r or F9 toggles   |   esc stops"
                 ).pack(side="left")

    def _build_reconstruction(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="x", pady=(16, 0))
        head = tk.Frame(wrap, bg=BG)
        head.pack(fill="x", pady=(0, 8))
        tk.Label(head, text="reconstruction", bg=BG, fg=FG_TEXT,
                 font=ui(13)).pack(side="left")
        self.recon_stats = tk.Label(head, bg=BG, fg=FG_LABEL, font=ui(10),
                                    text="what the model heard appears here")
        self.recon_stats.pack(side="left", padx=(10, 0))

        panel = RoundedPanel(wrap, BG_INPUT, radius=12, padding=10, bg=BG)
        panel.pack(fill="x")
        self.recon = tk.Text(
            panel.inner, height=4, width=88, bg=BG_INPUT, fg=FG_TEXT,
            font=mono(15), wrap="char", relief="flat", bd=0,
            highlightthickness=0, padx=8, pady=6, state="disabled")
        self.recon.pack(fill="x")

    def _build_detail(self, parent):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True, pady=(16, 0))
        tk.Label(wrap, text="per-key", bg=BG, fg=FG_TEXT,
                 font=ui(13)).pack(anchor="w", pady=(0, 8))
        panel = RoundedPanel(wrap, BG_PANEL, radius=12, padding=10, bg=BG)
        panel.pack(fill="both", expand=True)
        holder = tk.Frame(panel.inner, bg=BG_PANEL)
        holder.pack(fill="both", expand=True)
        self.detail = tk.Text(
            holder, height=8, bg=BG_PANEL, fg=FG_TEXT, font=mono(11),
            wrap="none", relief="flat", bd=0, highlightthickness=0,
            padx=8, pady=6, state="disabled")
        scroll = tk.Scrollbar(holder, command=self.detail.yview)
        self.detail.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.detail.pack(side="left", fill="both", expand=True)
        self.detail.tag_configure("ok", foreground=OK)
        self.detail.tag_configure("bad", foreground=FG_INCORRECT)
        self.detail.tag_configure("dim", foreground=FG_DIM)
        self.detail.tag_configure("warn", foreground=WARN)

    def _build_footer(self, parent):
        footer = tk.Frame(parent, bg=BG)
        footer.pack(fill="x", pady=(16, 0))
        holder = tk.Frame(footer, bg=BG)
        holder.pack(side="left")
        self.spinner = Spinner(holder, bg=BG)
        PillButton(footer, "infer from file...", self._infer_from_file, size=10,
                   bg=BG, padx=12, pady=6).pack(side="right")
        self.status_label = tk.Label(footer, bg=BG, fg=FG_LABEL, font=ui(11),
                                     anchor="w", justify="left",
                                     text="load a trained model to begin")
        self.status_label.pack(side="left", padx=(8, 0))

    # -- helpers ------------------------------------------------------------

    def _status(self, text, colour=None):
        self.status_label.configure(text=text, fg=colour or FG_LABEL)

    def _render_score_toggle(self):
        box = "[x]" if self.score_var.get() else "[ ]"
        self.score_toggle.configure(
            text=f"{box} score against what I type",
            fg=FG_TEXT if self.score_var.get() else FG_LABEL)

    def _toggle_score(self):
        if self.recording:
            return
        self.score_var.set(not self.score_var.get())
        self._render_score_toggle()

    def _short(self, path):
        return path if len(path) <= 40 else "..." + path[-37:]

    # -- model --------------------------------------------------------------

    def _choose_model(self):
        path = filedialog.askopenfilename(
            title="Trained model",
            initialdir=os.path.dirname(self.model_path) or ".",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")])
        if path:
            self._try_load(path, announce=True)

    def _try_load(self, path, announce=True):
        try:
            bundle = load_bundle(path)
        except FileNotFoundError:
            if announce:
                self._status(f"no model at {self._short(path)}", WARN)
            return
        except Exception as exc:
            self._status(f"could not load model: {exc}", FG_INCORRECT)
            return
        self.bundle = bundle
        self.model_path = os.path.abspath(path)
        self.sr = bundle.get("sample_rate", self.sr)
        n = len(bundle["label_encoder"].classes_)
        self.model_label.configure(text=f"{os.path.basename(path)}  ({n} keys)")
        if SessionRecorder.available():
            self.record_btn.set_enabled(True)
        self._status(f"model loaded - {n} keys known. record and type away.", OK)

    def _require_model(self):
        if self.bundle is None:
            self._status("load a trained model first", WARN)
            return False
        return True

    # -- recording ----------------------------------------------------------

    def _toggle_record(self):
        if self.recording:
            self._stop_recording()
        elif self._require_model():
            self._start_recording()

    def _start_recording(self):
        if self.recording or not self._require_model():
            return
        self.recorder = SessionRecorder(sr=self.sr, device=self.device)
        try:
            self.recorder.start()
        except Exception as exc:
            self.recorder = None
            self._status(f"could not start recording: {exc}", FG_INCORRECT)
            return
        self.recording = True
        self.record_btn.set_text("stop and read back")
        self.dot.itemconfigure(self._dot_item, fill=ACCENT)
        self.meter.clear()
        self._set_recon("", ())
        self._set_detail([("dim", "listening...\n")])
        mode = ("type the text and it will be scored" if self.score_var.get()
                else "pure listen - no ground truth")
        self._status(f"recording - {mode}", ACCENT)

    def _stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        recorder, self.recorder = self.recorder, None
        self.record_btn.set_text("start listening")
        self.record_btn.set_enabled(False)
        self.dot.itemconfigure(self._dot_item, fill=FG_DIM)
        self.meter.clear()

        capture = recorder.stop()
        if len(capture["audio"]) == 0:
            self.record_btn.set_enabled(True)
            self._status("stopped - no audio captured", WARN)
            return
        self._status("running the model over the recording...", FG_LABEL)
        self.spinner.start(side="left")
        scoring = self.score_var.get()
        threading.Thread(target=self._infer_worker, args=(capture, scoring),
                         daemon=True).start()

    def _infer_worker(self, capture, scoring):
        try:
            preds = infer_audio(capture["audio"], capture["sample_rate"],
                                self.bundle, params=self.params)
            report = (score_predictions(preds, capture["events"],
                                        capture["sample_rate"])
                      if scoring else None)
            self.results.put(("ok", (preds, report, capture)))
        except Exception as exc:                            # pragma: no cover
            self.results.put(("error", exc))

    def _drain_results(self):
        while True:
            try:
                kind, payload = self.results.get_nowait()
            except queue.Empty:
                break
            self.spinner.stop()
            self.record_btn.set_enabled(SessionRecorder.available()
                                        and self.bundle is not None)
            if kind == "error":
                self._status(f"inference failed: {payload}", FG_INCORRECT)
                continue
            preds, report, capture = payload
            self._present(preds, report, capture)
        self.root.after(80, self._drain_results)

    # -- presenting results -------------------------------------------------

    def _present(self, preds, report, capture):
        text = reconstruct_text(preds)
        confs = [p["confidence"] for p in preds]
        avg = sum(confs) / len(confs) if confs else 0.0
        self._set_recon(text, ())

        parts = [f"heard {len(preds)} keystrokes",
                 f"avg confidence {avg*100:.0f}%",
                 f"{capture['duration']:.1f}s"]
        colour = FG_LABEL
        if report is not None:
            parts.insert(0, f"accuracy {report['accuracy']*100:.0f}% "
                         f"({report['correct']}/{report['total']})")
            if report["missed"] or report["spurious"]:
                parts.append(f"{report['missed']} missed, "
                             f"{report['spurious']} spurious")
            colour = OK if report["accuracy"] >= 0.5 else WARN
        self.recon_stats.configure(text="   |   ".join(parts), fg=colour)
        self._status("done - see the reconstruction above", colour)

        self._set_detail(self._detail_rows(preds, report))

    def _detail_rows(self, preds, report):
        rows = []
        if report is not None:
            # Ground-truth view: one row per real keypress, so misses and
            # confusions show up in place.
            for i, m in enumerate(report["matched"]):
                true_disp = label_display(m["true"])
                if m["pred"] is None:
                    rows.append(("bad",
                                 f"{i:>3}  {true_disp:>4}  ->   (missed)\n"))
                    continue
                pred_disp = label_display(m["pred"])
                tag = "ok" if m["ok"] else "bad"
                mark = "ok " if m["ok"] else "MISS"
                rows.append((tag,
                             f"{i:>3}  {true_disp:>4}  ->  {pred_disp:<4}  "
                             f"{m['confidence']*100:5.1f}%  {mark}\n"))
            if report["spurious"]:
                rows.append(("warn",
                             f"\n+ {report['spurious']} spurious detection(s) "
                             "with no key behind them\n"))
        else:
            for pred in preds:
                alts = "  ".join(f"{label_display(n)}:{p*100:.0f}%"
                                 for n, p in pred["top"][1:3])
                tag = ("ok" if pred["confidence"] >= 0.6
                       else "warn" if pred["confidence"] >= 0.35 else "bad")
                rows.append((tag,
                             f"t={pred['t']:6.3f}s  "
                             f"{label_display(pred['label']):>4}  "
                             f"{pred['confidence']*100:5.1f}%   "
                             f"[{alts}]\n"))
        return rows or [("dim", "no keystrokes detected\n")]

    def _set_recon(self, text, tags):
        self.recon.configure(state="normal")
        self.recon.delete("1.0", "end")
        self.recon.insert("end", text or "-")
        self.recon.configure(state="disabled")

    def _set_detail(self, rows):
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        for tag, line in rows:
            self.detail.insert("end", line, (tag,))
        self.detail.configure(state="disabled")

    # -- input --------------------------------------------------------------

    def _on_keypress(self, event):
        if not self.recording or self.recorder is None:
            return
        if event.state & 0x0004:                # ctrl-chord: app shortcut
            return
        if event.keysym in IGNORED_KEYSYMS:
            return
        # Ground truth is only worth capturing when we mean to score against
        # it; in pure-listen mode the keyboard is just making sounds.
        if self.score_var.get():
            self.recorder.mark_key(event.keysym, event.char or "")

    def _tick(self):
        if self.recording and self.recorder is not None:
            elapsed = self.recorder.elapsed
            minutes, seconds = divmod(elapsed, 60)
            self.timer_label.configure(
                text=f"{int(minutes):02d}:{seconds:04.1f}")
            self.heard_label.configure(text=str(self.recorder.key_count))
            self.meter.push(self.recorder.level)
        self.root.after(60, self._tick)

    # -- file inference -----------------------------------------------------

    def _infer_from_file(self):
        if not self._require_model():
            return
        path = filedialog.askopenfilename(
            title="Recording to read back",
            filetypes=[("Audio", "*.wav *.mp3"), ("All files", "*.*")])
        if not path:
            return
        self._status(f"reading {os.path.basename(path)}...", FG_LABEL)
        self.spinner.start(side="left")
        threading.Thread(target=self._file_worker, args=(path,),
                         daemon=True).start()

    def _file_worker(self, path):
        try:
            sr = self.bundle.get("sample_rate", self.sr)
            audio, sr_ = load_audio(path, sr=sr)
            preds = infer_audio(audio, sr_, self.bundle, params=self.params)
            self.results.put(("ok", (preds, None,
                                     {"duration": len(audio) / float(sr_),
                                      "events": []})))
        except Exception as exc:                            # pragma: no cover
            self.results.put(("error", exc))


def launch_inference(model_path="keystroke_model.joblib", params=None,
                     sr=DEFAULT_SR, device=None):
    """Open the inference platform."""
    if _THEME_ERROR is not None:
        sys.exit("The inference UI needs this project's theme.py and anim.py "
                 f"alongside it ({_THEME_ERROR}).")
    root = tk.Tk()
    InferenceApp(root, model_path=model_path, params=params, sr=sr,
                 device=device)
    root.mainloop()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Annotate keystroke audio by typing, then train on it")
    parser.add_argument("--data_dir", type=str, default=None,
                        help=f"dataset folder (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--annotate", action="store_true",
                        help="open the annotation platform (the default)")
    parser.add_argument("--infer", action="store_true",
                        help="open the inference platform (record and read back)")
    parser.add_argument("--infer_file", type=str, metavar="CLIP",
                        help="reconstruct the typing in a recording, headless")
    parser.add_argument("--train", action="store_true",
                        help="train a model from --data_dir")
    parser.add_argument("--predict", type=str,
                        help="classify one .wav clip with a trained model")
    parser.add_argument("--resegment", type=str, metavar="DATA_DIR",
                        help="re-cut saved sessions with the current windows")
    parser.add_argument("--model_out", type=str, default="keystroke_model.joblib",
                        help="where to save/load the model")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR,
                        help="sample rate for recording and loading audio")
    parser.add_argument("--device", type=str, default=None,
                        help="input device name or index for recording")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help="clips per key the coverage grid aims for")
    parser.add_argument("--pre_ms", type=float, default=20.0,
                        help="clip milliseconds kept before the onset")
    parser.add_argument("--post_ms", type=float, default=180.0,
                        help="clip milliseconds kept after the onset")
    parser.add_argument("--min_gap_ms", type=float, default=80.0,
                        help="keys closer together than this are dropped")
    parser.add_argument("--keep_flagged", action="store_true",
                        help="also write dropped clips under _rejected/")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params = SegmentParams.from_args(args)
    data_dir = args.data_dir or DEFAULT_DATA_DIR

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)

    if args.predict:
        if not os.path.exists(args.model_out):
            sys.exit(f"Model file not found: {args.model_out}. Train first.")
        predict(args.predict, args.model_out)
    elif args.infer_file:
        infer_file(args.infer_file, args.model_out, params=params)
    elif args.resegment:
        resegment(args.resegment, params)
    elif args.infer:
        launch_inference(args.model_out, params=params, sr=args.sr,
                         device=device)
    # A bare --data_dir still trains, as it did before the annotator existed.
    elif args.train or (args.data_dir and not args.annotate):
        train(data_dir, args.model_out, sr=args.sr)
    else:
        launch_annotator(data_dir, params=params, target=args.target,
                         sr=args.sr, device=device)


if __name__ == "__main__":
    main()
