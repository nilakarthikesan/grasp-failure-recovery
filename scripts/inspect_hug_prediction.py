#!/usr/bin/env python
"""Entry point: inspect an official HUG grasp_pred/*.pkl output.

This reads a prediction the user already generated with HUG's released
inference code (in HUG's own Ubuntu/CUDA/Python 3.10 environment). It does not
import or run the HUG package.

Run from the repository root:

    python scripts/inspect_hug_prediction.py /path/to/grasp_pred/example.pkl
"""

from __future__ import annotations

from grasp_failure_prediction.integrations.hug import main

if __name__ == "__main__":
    main()
