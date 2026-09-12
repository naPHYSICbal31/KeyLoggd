"""Cutting one long typing run into enrollment samples.

The enrollment phrase tools produce one sample per short phrase: you type
nineteen characters, that is a sample, you do it again. The typing test
produces the opposite shape -- one unbroken run of several hundred
keystrokes -- so something has to decide where one sample ends and the next
begins.

That decision is here rather than in the UI, because it is a signal-processing
choice and not a presentation one: the pipeline reads a fixed window of
KEYS_PER_SAMPLE keystrokes per sample (see fft_features.FFT_LEN), so cutting
the run into windows of exactly that size is what makes every sample carry a
full spectrum instead of a mostly zero-padded one. A 30-second test at a
normal pace yields four or five samples, which is why the typing test can
reach a usable enrollment in a handful of runs.

Stdlib only, deliberately: the enroll view calls this on every finished test
and should not drag numpy in behind it.
"""

from datetime import datetime

#: Keystrokes per emitted sample. Mirrors fft_features.FFT_LEN -- the length
#: every signal is zero-padded or truncated to before its FFT. Kept as a
#: literal so this module stays import-light; the two must move together.
KEYS_PER_SAMPLE = 32

#: A trailing window shorter than this is dropped rather than zero-padded.
#: Half the window is the point past which padding, not typing, dominates the
#: spectrum -- such a sample would describe the padding more than the typist.
MIN_KEYS = KEYS_PER_SAMPLE // 2


def pair_keystrokes(events):
    """[(key, down_t, up_t), ...] in press order, from a raw event stream.

    events: [{"key", "type": "down"/"up", "t"}, ...] in chronological order.

    Each down is matched to the next up carrying the same key, so ordinary
    overlap -- releasing 'h' after 'e' is already down, which every fast
    typist does -- pairs correctly. A down with no matching up (the key was
    still held when the run ended) is dropped rather than guessed at.
    """
    strokes = []
    for i, ev in enumerate(events):
        if ev["type"] != "down":
            continue
        for later in events[i + 1:]:
            if later["type"] == "up" and later["key"] == ev["key"]:
                strokes.append((ev["key"], ev["t"], later["t"]))
                break
    return strokes


def split_into_samples(events, *, keys_per_sample=KEYS_PER_SAMPLE,
                       min_keys=MIN_KEYS, captured_at=None):
    """Cut a raw event stream into samples in the project's JSON schema.

    Returns a list of {"events", "phrase", "captured_at"} dicts -- the same
    shape the phrase-at-a-time capture widget emits, so both paths feed the
    enrollment file identically.

    Each sample's timestamps are rebased to its own first keydown. Only
    differences (dwell, flight) are ever read downstream, so the rebase
    changes no feature; it means a sample read on its own still starts at
    zero, like every other sample in the dataset.
    """
    strokes = pair_keystrokes(events)
    stamp = captured_at or datetime.now().isoformat(timespec="seconds")

    samples = []
    for start in range(0, len(strokes), keys_per_sample):
        window = strokes[start:start + keys_per_sample]
        if len(window) < min_keys:
            break
        t0 = window[0][1]
        flat = []
        for key, down, up in window:
            flat.append({"key": key, "type": "down", "t": round(down - t0, 6)})
            flat.append({"key": key, "type": "up", "t": round(up - t0, 6)})
        samples.append({
            "events": flat,
            "phrase": "".join(key for key, _, _ in window),
            "captured_at": stamp,
        })
    return samples
