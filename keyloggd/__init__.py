"""keyloggd - keystroke-behaviour capture, enrollment and identification.

Two halves:

    keyloggd.pipeline   the signal processing: raw keystroke events ->
                        dwell/flight signals -> FFT features -> templates ->
                        an identification / verification decision.
    keyloggd.ui         the Tkinter frontend built on top of it.

The pipeline half has no Tk dependency and the UI half owns no maths, so
either can be exercised without the other.
"""
