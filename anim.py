"""
anim.py

A tiny tween engine for the project's Tkinter frontends.

Tk has no animation primitives -- the usual approach is a chain of
`widget.after(...)` callbacks per effect, which drifts (each hop costs a few
ms more than requested) and gets hard to cancel cleanly. Instead, one
Animator owns a single ~60fps tick loop and drives every active tween from
real elapsed time (time.perf_counter), so effects stay in sync, survive a
slow frame without stretching, and can be cancelled or restarted by key.

Everything here is progress-based: a tween calls `on_frame(p)` with an eased
p in [0, 1] and lets the caller decide what p means -- a colour blend, a bar
height, a caret position, a character count.

Set KEYLOGGD_NO_ANIM=1 to run the UI with every effect applied instantly
(useful for screenshots, tests, and anyone who prefers no motion).
"""

import os
import time
import tkinter as tk

ENABLED = os.environ.get("KEYLOGGD_NO_ANIM", "").lower() not in ("1", "true", "yes")

FRAME_MS = 16  # ~60fps


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def linear(t):
    return t


def ease_out_cubic(t):
    """Fast start, soft landing. The default: reads as responsive."""
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t):
    return 4 * t ** 3 if t < 0.5 else 1.0 - ((-2 * t + 2) ** 3) / 2


def ease_out_back(t, overshoot=1.7):
    """Overshoots slightly then settles -- a subtle 'pop'."""
    c3 = overshoot + 1.0
    return 1.0 + c3 * (t - 1) ** 3 + overshoot * (t - 1) ** 2


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _parse(colour):
    colour = colour.lstrip("#")
    return int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16)


def blend(start, end, amount):
    """Interpolate between two #rrggbb colours; amount 0 -> start, 1 -> end."""
    amount = max(0.0, min(1.0, amount))
    r1, g1, b1 = _parse(start)
    r2, g2, b2 = _parse(end)
    return "#%02x%02x%02x" % (
        round(r1 + (r2 - r1) * amount),
        round(g1 + (g2 - g1) * amount),
        round(b1 + (b2 - b1) * amount),
    )


def is_hex_colour(value):
    if not isinstance(value, str) or not value.startswith("#") or len(value) != 7:
        return False
    try:
        _parse(value)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Animator
# ---------------------------------------------------------------------------

class Animator:
    """Drives every active tween from one after() loop on `widget`."""

    def __init__(self, widget: tk.Misc, interval=FRAME_MS):
        self.widget = widget
        self.interval = interval
        self._tweens = {}
        self._job = None
        self._seq = 0

    # -- scheduling ---------------------------------------------------------

    def tween(self, duration, on_frame, *, easing=ease_out_cubic, on_done=None,
              key=None, delay=0):
        """Run `on_frame(eased_p)` for `duration` ms, then `on_done()`.

        `key` makes a tween replaceable: scheduling the same key again drops
        the in-flight one (snapping it to its end state) and starts over, so
        rapid retriggers -- a caret chasing fast keystrokes, a hover in and
        out -- never leave two tweens fighting over the same property.
        `delay` (ms) holds at p=0 first, which is what staggers a group.
        """
        if key is None:
            self._seq += 1
            key = f"__anon{self._seq}"
        self.cancel(key, finish=True)

        if not ENABLED:
            on_frame(1.0)
            if on_done:
                on_done()
            return key

        on_frame(0.0)  # start from the initial state, even while delayed
        self._tweens[key] = {
            "start": time.perf_counter() + delay / 1000.0,
            "duration": max(duration, 1) / 1000.0,
            "frame": on_frame,
            "easing": easing,
            "done": on_done,
        }
        self._ensure_running()
        return key

    def repeat(self, duration, on_frame, *, easing=linear, key=None, delay=0):
        """A tween that restarts forever; cancel by key to stop it."""
        if not ENABLED:
            # with motion off there is no loop to run: draw the resting state
            # once, or the restart callback would recurse without end
            on_frame(1.0)
            return key

        def again():
            self.repeat(duration, on_frame, easing=easing, key=key)

        return self.tween(duration, on_frame, easing=easing, on_done=again,
                          key=key, delay=delay)

    # -- cancelling ---------------------------------------------------------

    def cancel(self, key, *, finish=False):
        """Drop a tween. finish=True snaps it to its end state first, which
        avoids leaving a half-faded colour or half-grown bar on screen."""
        tween = self._tweens.pop(key, None)
        if tween is None:
            return False
        if finish:
            try:
                tween["frame"](1.0)
            except tk.TclError:
                pass
        return True

    def cancel_all(self, *, finish=False):
        for key in list(self._tweens):
            self.cancel(key, finish=finish)

    def running(self, key):
        return key in self._tweens

    def cancel_matching(self, prefixes, *, finish=False):
        """Drop every tween whose key starts with one of `prefixes`."""
        for key in list(self._tweens):
            if key.startswith(prefixes):
                self.cancel(key, finish=finish)

    # -- the loop -----------------------------------------------------------

    def _ensure_running(self):
        if self._job is None and self._tweens:
            self._job = self.widget.after(self.interval, self._tick)

    def _tick(self):
        self._job = None
        now = time.perf_counter()

        for key, tween in list(self._tweens.items()):
            if now < tween["start"]:
                continue  # still in its stagger delay
            raw = min(1.0, (now - tween["start"]) / tween["duration"])
            try:
                tween["frame"](tween["easing"](raw))
            except tk.TclError:
                self._tweens.pop(key, None)  # widget went away mid-flight
                continue
            if raw >= 1.0:
                self._tweens.pop(key, None)
                if tween["done"]:
                    try:
                        tween["done"]()
                    except tk.TclError:
                        pass

        try:
            self._ensure_running()
        except tk.TclError:
            self._tweens.clear()


# ---------------------------------------------------------------------------
# Composite effects
# ---------------------------------------------------------------------------

# Widget options worth fading, per option name.
_FADE_OPTIONS = ("foreground", "background")


def collect_fade_targets(widget, skip=()):
    """Every (widget, option, target_colour) pair in a subtree worth fading.

    Snapshot this while the widgets still hold their design colours -- right
    after building them. Reading it later can catch a colour that another
    tween is midway through changing, and the fade would then treat that
    transient value as the colour to settle on.
    """
    targets = []
    stack = [widget]
    while stack:
        node = stack.pop()
        if node in skip:
            continue
        for option in _FADE_OPTIONS:
            try:
                value = str(node.cget(option))
            except (tk.TclError, AttributeError):
                continue
            if is_hex_colour(value):
                targets.append((node, option, value))
        stack.extend(node.winfo_children())
    return targets


def fade_in_subtree(animator: Animator, widget, from_colour, *, duration=280,
                    delay=0, key=None, skip=(), targets=None):
    """Appear-in effect: every colour in the subtree starts at `from_colour`
    (the surface the block sits on) and resolves to its real value, so the
    block emerges from the background instead of popping into place.

    Pass `targets` from an earlier collect_fade_targets() call to fade towards
    the widgets' design colours rather than whatever they happen to show now.
    """
    if targets is None:
        targets = collect_fade_targets(widget, skip=skip)
    if not targets:
        return None

    def frame(p):
        for node, option, target in targets:
            node.configure(**{option: blend(from_colour, target, p)})

    return animator.tween(duration, frame, delay=delay,
                          key=key or f"fade{id(widget)}")


def reveal_text(animator: Animator, label, text, *, duration=280, key=None,
                colour=None, from_colour=None):
    """Type `text` out one character at a time -- a keystroke-project-native
    way to land a verdict -- optionally fading the colour in alongside."""
    def frame(p):
        shown = max(1, round(len(text) * p)) if text else 0
        options = {"text": text[:shown]}
        if colour and from_colour:
            options["fg"] = blend(from_colour, colour, min(1.0, p * 1.6))
        label.configure(**options)

    return animator.tween(duration, frame, easing=linear,
                          key=key or f"reveal{id(label)}")


def stagger(index, count, spread=0.55):
    """Per-item (start, span) progress window for a staggered group.

    Given the group's overall progress p, item i animates over
    [start, start + span] -- so item 0 leads and the last item trails by
    `spread` of the total duration.
    """
    if count <= 1:
        return 0.0, 1.0
    start = (index / (count - 1)) * spread
    return start, 1.0 - spread


def local_progress(p, index, count, spread=0.55):
    """The 0..1 progress of item `index` within a staggered group at time p."""
    start, span = stagger(index, count, spread)
    return max(0.0, min(1.0, (p - start) / span))
