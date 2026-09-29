#!/usr/bin/env python
"""Entry point: validate the Part I robot-training stack.

Usage:
    MUJOCO_GL=cgl python scripts/validate_stack.py
    MUJOCO_GL=cgl python scripts/validate_stack.py --no-sim --no-lerobot
"""

from grasp_failure_prediction.part1.stack import main

if __name__ == "__main__":
    raise SystemExit(main())
