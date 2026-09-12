#!/usr/bin/env python3
"""keyloggd - keystroke behaviour capture, enrollment and identification.

    python main.py                  the app (splash, then enroll/identify/explain)
    python main.py --no-splash      skip the splash
    python main.py --data-dir DIR   use a different enrollment folder

The typing test can also be run on its own:

    python -m keyloggd.ui.typing_test

and the pipeline has its own entry points:

    python -m keyloggd.pipeline.classifier        identification + EER report
    python -m keyloggd.pipeline.identify_sample   guess who typed a JSON file
    python -m tools.synthetic_data_generator      generate synthetic users
"""

from keyloggd.ui.app import main

if __name__ == "__main__":
    main()
