"""CLI for retargeting one saved HUG prediction and viewing it in MuJoCo."""

from __future__ import annotations

import argparse

from grasp_failure_prediction.integrations.hug import load_hug_prediction

from .pose_validation import ShadowPoseValidator, format_validation
from .retargeting import ShadowHandRetargeter, default_dex_urdf_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", help="trusted HUG grasp_pred/*.pkl file")
    parser.add_argument(
        "--viewer", action="store_true", help="open the native MuJoCo viewer"
    )
    args = parser.parse_args()

    grasp = load_hug_prediction(args.prediction)
    urdf_root = default_dex_urdf_root()
    retargeter = ShadowHandRetargeter(urdf_root)
    pose = retargeter.retarget(grasp)
    validator = ShadowPoseValidator(urdf_root)
    result = validator.validate(pose)
    print(format_validation(result))
    if args.viewer:
        validator.launch_viewer(pose)


if __name__ == "__main__":
    main()
