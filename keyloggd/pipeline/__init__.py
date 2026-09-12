"""The keystroke-dynamics pipeline, in the order the data flows through it.

    signal_construction   (key, down/up, t) events  ->  dwell[n], flight[n]
    fft_features          those signals             ->  a 14-number vector
    denoise               optional low-pass cleanup of a signal
    classifier            vectors -> templates, kNN, EER evaluation
    identify_sample       enrolled set + one unknown sample -> a verdict

Nothing here imports tkinter; numpy is the only third-party dependency.
"""
