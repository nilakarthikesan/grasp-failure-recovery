"""Explicit opt-in UniDexGrasp proposal worker in the external CUDA runtime.

Uses only IPDF + Glow proposal sampling. It does not invoke upstream eval.py,
dataset loaders, quality evaluation, execution policies, or physics rollouts.
No work happens on import. Checkpoints and the source MJCF are caller supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from grasp_failure_prediction.evaluation.schema import canonical_json
from grasp_failure_prediction.proposals.artifacts import file_sha256, load_request
from grasp_failure_prediction.proposals.registry import load_provider_registry
from grasp_failure_prediction.proposals.schema import RigidTransform
from grasp_failure_prediction.proposals.unidexgrasp import (
    SOURCE_COMMIT, SOURCE_JOINT_NAMES, SOURCE_REPOSITORY,
    UniDexGraspProposalProvider, _child, _prepare_source, _verify_file,
)


def read_worker_input(path: Path, request):
    """Validate caller-prepared physical point coordinates, without sampling."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version", "object_id", "geometry_content_hash", "object_scale",
        "object_to_input", "point_cloud_path", "point_cloud_sha256", "point_cloud_units",
        "point_cloud_frame", "plane_input",
    }
    if set(raw) != expected_keys or raw["schema_version"] != "unidexgrasp_object_input_v1":
        raise ValueError("worker requires a strict unidexgrasp_object_input_v1 manifest")
    if raw["object_id"] != request.object.id or raw["geometry_content_hash"] != request.object.geometry_content_hash:
        raise ValueError("UniDex object/geometry binding differs from request")
    if raw["point_cloud_units"] != "m" or raw["point_cloud_frame"] != "input_point_cloud":
        raise ValueError("UniDex point coordinates must be physical meters in the input point-cloud frame")
    scale = raw["object_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("object_scale must be finite and positive (geometry already scaled)")
    transform = RigidTransform.model_validate(raw["object_to_input"])
    plane = np.asarray(raw["plane_input"])
    if plane.shape != (4,) or plane.dtype.kind not in "fiu" or not np.isfinite(plane).all() or not np.isclose(np.linalg.norm(plane[:3]), 1., atol=1e-6):
        raise ValueError("plane_input must contain a unit normal and finite metric offset")
    points_file = _child(path.parent, raw["point_cloud_path"], "point cloud")
    _verify_file(points_file, raw["point_cloud_sha256"], "point cloud")
    points = np.load(points_file, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1024 or points.dtype.kind not in "fiu" or not np.isfinite(points).all():
        raise ValueError("caller-prepared point cloud must contain at least 1024 finite XYZ points")
    return raw, transform, plane.astype(np.float32), points.astype(np.float32)


def _assert_runtime(root: Path):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(f"UniDex runtime must be pinned to {SOURCE_COMMIT}")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "dexgrasp_generation"], cwd=root, check=True)
    generation = root / "dexgrasp_generation"
    if not (generation / "network/models/model.py").is_file():
        raise ValueError("runtime has no UniDex generation model source")
    return generation


def _assert_official_modules(generation: Path):
    """Reject cached/import-shadowed upstream Python modules before loading weights."""
    for name, module in tuple(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in ("network", "utils", "datasets")):
            continue
        location = getattr(module, "__file__", None)
        if location is not None and not Path(location).resolve().is_relative_to(generation):
            raise ValueError(f"UniDex module {name} was imported outside its pinned generation checkout")
        for path in getattr(module, "__path__", ()):
            if not Path(path).resolve().is_relative_to(generation):
                raise ValueError(f"UniDex package {name} resolves outside its pinned generation checkout")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--unidexgrasp-root", type=Path, required=True)
    parser.add_argument("--rotation-checkpoint", type=Path, required=True)
    parser.add_argument("--rotation-checkpoint-sha256", required=True)
    parser.add_argument("--pose-checkpoint", type=Path, required=True)
    parser.add_argument("--pose-checkpoint-sha256", required=True)
    parser.add_argument("--source-hand-mjcf", type=Path)
    parser.add_argument("--source-hand-mjcf-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    request = load_request(args.request)
    if request.provider.id != "unidexgrasp_shadow_v1" or request.provider_input_path is None or request.observation_path is not None or request.provider_inputs:
        raise ValueError("worker requires a UniDex request with provider_input_path as its sole input channel")
    if request.seed + request.sample_count - 1 > 2**32 - 1:
        raise ValueError("sample seeds exceed unsigned 32-bit range")
    input_file = (project / request.provider_input_path).resolve()
    if not input_file.is_relative_to(project):
        raise ValueError("worker input escapes project root")
    raw, transform, plane, points = read_worker_input(input_file, request)
    source_root = args.unidexgrasp_root.resolve()
    generation = _assert_runtime(source_root)
    source = (args.source_hand_mjcf or generation / "data/mjcf/shadow_hand.xml").resolve()
    source_hash = args.source_hand_mjcf_sha256
    with tempfile.TemporaryDirectory(prefix="unidex-worker-asset-check-") as temp:
        _prepare_source(source, Path(temp) / "source.xml", source_hash)
    checkpoints = {"rotation": args.rotation_checkpoint.resolve(), "pose": args.pose_checkpoint.resolve()}
    checkpoint_hashes = {"rotation": args.rotation_checkpoint_sha256, "pose": args.pose_checkpoint_sha256}
    for key, checkpoint in checkpoints.items():
        _verify_file(checkpoint, checkpoint_hashes[key], f"{key} checkpoint")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("worker output directory must be empty")

    # The GPU runtime imports are intentionally after all request/asset checks.
    # Loading model checkpoints is authorized only when this worker is invoked.
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    from grasp_failure_prediction.evaluation.schema import canonical_sha256
    sys.path.insert(0, str(generation))
    sys.path.insert(0, str(generation / "network"))
    os.chdir(generation)
    from network.models.model import IPDFModel, GlowModel
    from utils.hand_model import add_rotation_to_hand_pose
    _assert_official_modules(generation)
    with initialize_config_dir(version_base=None, config_dir=str(generation / "configs")):
        rotation_cfg = compose(config_name="ipdf_config")
        pose_cfg = compose(config_name="glow_config")
        with open_dict(rotation_cfg):
            rotation_cfg["device"] = args.device
        with open_dict(pose_cfg):
            pose_cfg["device"] = args.device
            pose_cfg["model"]["joint_training"] = False
        config_hashes = {
            "rotation": canonical_sha256(OmegaConf.to_container(rotation_cfg, resolve=True)),
            "pose": canonical_sha256(OmegaConf.to_container(pose_cfg, resolve=True)),
        }
        torch.manual_seed(request.seed)
        rotation_model = IPDFModel(rotation_cfg).to(args.device).eval()
        pose_model = GlowModel(pose_cfg).to(args.device).eval()
        for model, key in ((rotation_model, "rotation"), (pose_model, "pose")):
            # Official checkpoints contain a nested model state dict. They
            # remain within this explicitly invoked, caller-trusted runtime.
            state = torch.load(str(checkpoints[key]), map_location=args.device)
            if not isinstance(state, dict) or "model" not in state:
                raise ValueError(f"{key} checkpoint lacks the official model state dictionary")
            model.load_state_dict(state["model"], strict=True)
        point_tensor = torch.from_numpy(points).unsqueeze(0).to(args.device)
        plane_tensor = torch.from_numpy(plane).unsqueeze(0).to(args.device)
        poses = []
        for index in range(request.sample_count):
            seed = request.seed + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                data = {"obj_pc": point_tensor, "plane": plane_tensor}
                rotation_model.set_data(data)
                rotation_model.test(no_eval=True)
                data.update(rotation_model.pred_dict)
                pose_model.set_data(data)
                pose_model.test(no_eval=True)
                predicted = pose_model.pred_dict
                canonical = torch.cat((predicted["canon_translation"], torch.zeros_like(predicted["canon_translation"]), predicted["hand_qpos"]), dim=-1)
                actual = add_rotation_to_hand_pose(canonical, data["sampled_rotation"])
                poses.append(actual.detach().cpu().numpy()[0])
    output.mkdir(parents=True, exist_ok=True)
    native = output / "_unidexgrasp_export"
    native.mkdir()
    model_output = native / "model_output.npz"
    np.savez_compressed(model_output, hand_pose=np.asarray(poses, dtype=np.float32))
    copied_source = native / "source_hand.xml"
    shutil.copy2(source, copied_source)
    report = {
        "schema_version": "unidexgrasp_inference_report_v1", "classification": "fresh_model_inference",
        "source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
        "object_id": request.object.id, "geometry_content_hash": request.object.geometry_content_hash,
        "object_scale": raw["object_scale"], "object_to_input": transform.model_dump(mode="json"),
        "sample_count": request.sample_count, "seeds": [request.seed + index for index in range(request.sample_count)],
        "checkpoint_sha256": checkpoint_hashes, "config_sha256": config_hashes,
        "source_hand_mjcf_sha256": source_hash, "output_sha256": file_sha256(model_output),
        "input_manifest_sha256": file_sha256(input_file), "point_cloud_sha256": raw["point_cloud_sha256"],
        "sampling_variant": "ipdf_plus_glow_without_tta", "tta_applied": False,
    }
    report_file = native / "inference_report.json"
    report_file.write_bytes(canonical_json(report) + b"\n")
    manifest = {
        "schema_version": "unidexgrasp_native_bundle_v1", "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT, "object_id": request.object.id,
        "geometry_content_hash": request.object.geometry_content_hash, "object_scale": raw["object_scale"],
        "object_to_input": transform.model_dump(mode="json"), "pose_frame": "input_point_cloud",
        "translation_units": "m", "rotation_representation": "axis_angle", "joint_units": "rad",
        "joint_names": list(SOURCE_JOINT_NAMES), "pose_field": "hand_pose",
        "output_path": "model_output.npz", "output_sha256": file_sha256(model_output),
        "report_path": "inference_report.json", "report_sha256": file_sha256(report_file),
        "source_hand_mjcf_path": "source_hand.xml", "source_hand_mjcf_sha256": source_hash,
    }
    bundle = native / "native_manifest.json"
    bundle.write_bytes(canonical_json(manifest) + b"\n")
    provider = UniDexGraspProposalProvider(load_provider_registry().resolve(request.provider.id), project_root=project)
    paths = provider.generate_from_bundle(request, output, bundle, binding_path=input_file)
    print(json.dumps({"proposal_paths": [path.relative_to(output).as_posix() for path in paths]}))


if __name__ == "__main__":
    main()
