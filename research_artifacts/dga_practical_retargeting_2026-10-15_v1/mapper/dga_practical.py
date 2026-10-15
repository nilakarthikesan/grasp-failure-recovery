"""Static DGA-to-Shadow reference for the practical retargeting protocol.

This module defines an explicit joint-limit projection *after* native decoding.
It never repairs source-invalid data or assigns physical/fidelity acceptance.
Geometry-aware corrections require the separate frozen study's validation gate.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from . import dexgrasp_anything as dga

PROTOCOL_ID = "dga_shadow_practical_static_v1"
BRANCHES = {"index": "FF", "middle": "MF", "ring": "RF", "little": "LF", "thumb": "TH"}
NUMERICAL_FRAME_ATOL = 1e-9  # arithmetic consistency, not a grasp-fidelity budget


def _source(native):
    if native.source_hand_sha256 != dga.SOURCE_URDF_SHA256:
        raise dga.DGAContractError("source_hand_mismatch", "Pinned source required")
    source = dga.DGAHandTree(native.source_hand_urdf, dga.SOURCE_URDF_SHA256)
    local = source.forward(native.joint_targets_rad)  # source limits, before any projection
    root = dga.proper_transform(native.object_from_forearm)
    if set(native.object_from_links) != set(local):
        raise dga.DGAContractError("native_link_set_mismatch", "Exact source link set required")
    for name, frame in local.items():
        actual = dga.proper_transform(native.object_from_links[name])
        if not np.allclose(actual, root @ frame, atol=NUMERICAL_FRAME_ATOL, rtol=0):
            raise dga.DGAContractError("native_frame_inconsistent", name)
    palm = dga.proper_transform(native.object_from_palm)
    if not np.allclose(palm, root @ local["palm"], atol=NUMERICAL_FRAME_ATOL, rtol=0):
        raise dga.DGAContractError("native_frame_inconsistent", "palm")
    return palm


def _target(native, target_hand_urdf):
    target = dga.DGAHandTree(Path(target_hand_urdf), dga.TARGET_URDF_SHA256)
    for name in ("WRJ2", "WRJ1"):
        lo, hi = target.joints[name][4]
        if not lo <= native.joint_targets_rad[name] <= hi:
            raise dga.DGAContractError("practical_wrist_unsupported", "Native wrist outside target domain")
    return target


def validate_practical_pose(native, *, joint_targets_rad, object_from_palm,
                            target_hand_urdf: Path):
    """Validate representation and fixed-branch rules; no geometric acceptance.

    Returns the independently reconstructible target root. Supplied transforms
    never override the canonical native palm after the consistency check.
    """
    palm = _source(native)
    target = _target(native, target_hand_urdf)
    frames = target.forward(joint_targets_rad)
    supplied = dga.proper_transform(object_from_palm)
    if not np.array_equal(supplied, palm):
        raise dga.DGAContractError("practical_palm_changed", "Practical v1 fixes native palm exactly")
    for name in ("WRJ2", "WRJ1"):
        if joint_targets_rad[name] != native.joint_targets_rad[name]:
            raise dga.DGAContractError("practical_wrist_changed", name)
    preserved = []
    for part, prefix in BRANCHES.items():
        names = [n for n in dga.SOURCE_JOINT_NAMES if n.startswith(prefix)]
        if all(target.joints[n][4][0] <= native.joint_targets_rad[n] <= target.joints[n][4][1] for n in names):
            if any(joint_targets_rad[n] != native.joint_targets_rad[n] for n in names):
                raise dga.DGAContractError("practical_legal_branch_changed", part)
            preserved.append(part)
    root = palm @ np.linalg.inv(frames["palm"])
    dga.proper_transform(root)
    if not np.allclose(root @ frames["palm"], palm, atol=NUMERICAL_FRAME_ATOL, rtol=0):
        raise dga.DGAContractError("practical_root_chain_invalid", "Target palm reconstruction")
    return {"object_from_target_root": root.tolist(), "preserved_legal_branches": preserved,
            "structural_status": "PASS", "geometry_status": "NOT_ASSESSED",
            "physical_status": "NOT_ASSESSED", "execution_eligible": False}


def build_practical_reference(native, *, target_hand_urdf: Path):
    """Unique Euclidean projection of finger angles onto the literal target box.

    This minimizes angular displacement only, not native geometric distortion.
    Every changed angle is reported. Native pose fields remain untouched.
    """
    palm = _source(native)
    target = _target(native, target_hand_urdf)
    q = dict(native.joint_targets_rad)
    changes = {}
    for name in dga.SOURCE_JOINT_NAMES:
        if name.startswith("WRJ"):
            continue
        lo, hi = target.joints[name][4]
        projected = float(min(max(q[name], lo), hi))
        if projected != q[name]:
            changes[name] = {"source_rad": float(q[name]), "target_rad": projected,
                             "delta_rad": projected - float(q[name]), "reason": "target_interval_projection"}
        q[name] = projected
    checked = validate_practical_pose(native, joint_targets_rad=q, object_from_palm=palm,
                                      target_hand_urdf=target_hand_urdf)
    return {"schema_version": "dga_practical_reference_v1", "protocol_id": PROTOCOL_ID,
            "mapping_stage": "explicit_joint_projection_after_source_validation",
            "source_hand_sha256": native.source_hand_sha256, "target_hand_sha256": target.sha256,
            "source_joint_targets_rad": dict(native.joint_targets_rad),
            "joint_targets_rad": q, "joint_changes": changes,
            "object_from_palm": palm.tolist(), **checked}
