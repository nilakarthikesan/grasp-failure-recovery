#!/usr/bin/env python
"""Entry point: inspect AdroitHandRelocate-v1 without training or acting.

Run from the repository root:

    python scripts/inspect_adroit.py
"""

from __future__ import annotations

from grasp_failure_prediction.environments.adroit import main

if __name__ == "__main__":
    main()
