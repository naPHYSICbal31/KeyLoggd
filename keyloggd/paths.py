"""Where the project keeps things on disk.

Four modules used to each recompute the enrollment folder from their own
__file__, which quietly breaks the moment one of them moves. They all import
it from here instead, so moving a module is a move and nothing more.
"""

import os

#: the repository root -- this file lives at <root>/keyloggd/paths.py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")

#: enrollment folder: one <user_id>.json per enrolled typist, in the schema
#: signal_construction.py reads. Real captures and synthetic users share it.
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")

#: multi-phrase synthetic users, for measuring free-text (cross-phrase)
#: performance rather than fixed-phrase performance.
SYNTHETIC_MULTI_DIR = os.path.join(DATA_DIR, "synthetic_multi")
