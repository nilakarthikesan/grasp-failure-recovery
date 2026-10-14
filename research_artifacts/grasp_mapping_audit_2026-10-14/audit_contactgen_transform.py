"""Static Shadow FK composition audit, algebra only.

No ContactGen inference/output, retargeting optimization, model-generated poses,
physics stepping, rollout, or evaluation data is involved. Predefined wrist
angles exercise the adapter's frame composition on the existing Shadow URDF.
The URDF is compiled in memory; only the requested audit JSON is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from grasp_failure_prediction.evaluation.pose_validation import _link_positions
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT as A
from grasp_failure_prediction.proposals.contactgen import (
    ManoLandmarkInput, axis_angle_rotation, contactgen_object_palm_pose,
    rigid_transform_matrix,
)


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("contactgen_transform_review.json"))
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output.resolve()
    audit_directory = Path(__file__).resolve().parent
    if not output.is_relative_to(audit_directory):
        raise ValueError("Audit output must remain in this /private/tmp audit directory")

    # mj_forward computes geometry/FK. All dynamics stepping entry points are
    # blocked before compiling or evaluating the existing hand asset.
    step_calls = []
    def forbidden_step(*args, **kwargs):
        step_calls.append(True)
        raise RuntimeError("Physics stepping is prohibited in this algebra-only audit")
    guarded = []
    for name in ("mj_step", "mj_step1", "mj_step2"):
        if hasattr(mujoco, name):
            setattr(mujoco, name, forbidden_step)
            guarded.append(name)

    source = project / "third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf"
    xml = ET.parse(source)
    for mesh in xml.getroot().findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    extension = ET.Element("mujoco")
    ET.SubElement(extension, "compiler", {"strippath": "false"})
    xml.getroot().insert(0, extension)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
    data = mujoco.MjData(model)
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    if palm_id < 0:
        raise ValueError("Existing Shadow URDF has no palm body")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]

    # Arbitrary predefined rigid transform; this is NOT a ContactGen prediction.
    mano_frame = np.eye(4)
    mano_frame[:3, :3] = axis_angle_rotation((0.4, -0.2, 0.3))
    mano_frame[:3, 3] = np.array((0.12, -0.08, 0.2))
    checks = []
    for wrist_angles in ((0.0, 0.0), (0.15, -0.2)):
        data.qpos[:] = 0.0
        data.qpos[names.index("WRJ1")] = wrist_angles[0]
        data.qpos[names.index("WRJ2")] = wrist_angles[1]
        mujoco.mj_forward(model, data)
        achieved_base = _link_positions(model, data, source) - data.xpos[palm_id]
        base_palm_rotation = data.xmat[palm_id].reshape(3, 3).copy()
        expected_object = (
            achieved_base @ (mano_frame[:3, :3] @ A.T).T + mano_frame[:3, 3]
        )
        validator = SimpleNamespace(model=model, data=data, set_pose=lambda pose: None)
        # No human keypoints are used: the helper under audit consumes only
        # this transform and the existing robot FK palm rotation.
        input_frame = ManoLandmarkInput(np.zeros((21, 3)), mano_frame)
        palm_pose = rigid_transform_matrix(
            contactgen_object_palm_pose(input_frame, object(), validator)
        )
        achieved_palm = achieved_base @ base_palm_rotation
        reconstructed_object = (
            achieved_palm @ palm_pose[:3, :3].T + palm_pose[:3, 3]
        )
        residual = float(np.max(np.linalg.norm(reconstructed_object - expected_object, axis=1)))
        checks.append({
            "predefined_wrist_angles_rad": {"WRJ1": wrist_angles[0], "WRJ2": wrist_angles[1]},
            "other_joint_values": "all zero",
            "task_points": "five URDF fingertips and five middle links",
            "maximum_object_frame_composition_residual_m": residual,
            "object_palm_rotation_determinant": float(np.linalg.det(palm_pose[:3, :3])),
            "algebra_check_passed": bool(residual < 1e-12),
        })
    report = {
        "schema_version": "contactgen_transform_audit_v1",
        "audit_date": "2026-10-14",
        "scope": "algebra-only: existing Shadow URDF static FK and adapter palm-frame composition",
        "real_contactgen_output_used": False,
        "contactgen_inference_run": False,
        "retargeting_optimization_run": False,
        "physics_stepping_run": False,
        "evaluation_data_generated": False,
        "stepping_guard_entry_points": guarded,
        "stepping_calls_attempted": len(step_calls),
        "joint_count": int(model.njnt),
        "project_root": str(project),
        "project_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        "shadow_urdf": {"path": str(source), "sha256": digest(source)},
        "adapter": {"path": str(project / "src/grasp_failure_prediction/proposals/contactgen.py"), "sha256": digest(project / "src/grasp_failure_prediction/proposals/contactgen.py")},
        "audit_script_sha256": digest(Path(__file__).resolve()),
        "runtime": {
            "executable": sys.executable, "python_version": sys.version,
            "numpy_version": np.__version__, "mujoco_version": mujoco.__version__,
            "pythonpath": os.environ.get("PYTHONPATH"),
        },
        "checks": checks,
        "limitations": [
            "This measures transform composition only, not ContactGen MANO-to-Shadow retargeting accuracy.",
            "No native human grasp, object mesh, contact map, contact preservation, penetration or lift stability was tested.",
        ],
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "checks_passed": all(item["algebra_check_passed"] for item in checks), "stepping_calls_attempted": len(step_calls)}))


if __name__ == "__main__":
    main()
