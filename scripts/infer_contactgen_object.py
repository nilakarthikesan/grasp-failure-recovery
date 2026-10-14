"""Official ContactGen + MANO refinement worker for --external-command.

This entry point requires an external, clean pinned ContactGen checkout, CUDA
runtime, checkpoints and licensed right MANO assets. It downloads nothing and
runs only when explicitly invoked. --help does not import Torch/ContactGen.
The input file is contactgen_object_input_v1 JSON via provider_input_path (or a
provider_input role). The output is canonical sample_*/proposal.json with a
hash-auditable refined-MANO bundle per sample; this script never runs physics.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
import random
import re
import subprocess
import sys

from grasp_failure_prediction.evaluation.schema import SHA256_PATTERN
from grasp_failure_prediction.proposals.artifacts import file_sha256, load_request
from grasp_failure_prediction.proposals.contactgen import (
    CONTACTGEN_PROVIDER_ID, CONTACTGEN_SOURCE_REVISION, ContactGenNativeBundle,
    ContactGenObjectInput, ContactGenOptimizedSample, _input_path,
    convert_contactgen_bundle, validate_contactgen_mesh,
)
from grasp_failure_prediction.proposals.registry import load_provider_registry


def verify_contactgen_source(root: Path) -> None:
    """Accept the pinned upstream code, with no modified tracked files."""
    if not root.is_dir():
        raise FileNotFoundError(f"ContactGen checkout does not exist: {root}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
    ).strip()
    if revision != CONTACTGEN_SOURCE_REVISION:
        raise ValueError(f"ContactGen source must be pinned to {CONTACTGEN_SOURCE_REVISION}")
    changed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=root, check=False,
    )
    if changed.returncode != 0:
        raise ValueError("ContactGen checkout has modified tracked files; provenance requires clean pinned source")
    required = (
        "contactgen/model.py", "contactgen/contact/contact_optimizer.py",
        "contactgen/configs/default.yaml", "contactgen/hand_sdf/config.yaml",
        "manopth/manolayer.py",
    )
    for relative in required:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"ContactGen runtime source is missing: {relative}")


def _official_module(name: str, root: Path):
    module = importlib.import_module(name)
    location = Path(module.__file__).resolve()
    if not location.is_relative_to(root):
        raise ValueError(f"{name} was imported from {location}, outside pinned ContactGen checkout")
    return module


def _file(path: Path, role: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ContactGen {role} does not exist: {resolved}")
    return resolved


def verify_runtime_asset(path: Path, expected_sha256: str, role: str) -> Path:
    """Verify the caller's exact asset pin before any executable pickle load."""
    resolved = _file(path, role)
    if re.fullmatch(SHA256_PATTERN, expected_sha256) is None:
        raise ValueError(f"ContactGen {role} expected hash must be sha256:<64 lowercase hex digits>")
    if file_sha256(resolved) != expected_sha256:
        raise ValueError(f"ContactGen {role} hash does not match the supplied asset pin")
    return resolved


def run_worker(args) -> list[Path]:
    project = args.project_root.resolve()
    request = load_request(args.request)
    if request.provider.id != CONTACTGEN_PROVIDER_ID:
        raise ValueError(f"ContactGen worker requires provider {CONTACTGEN_PROVIDER_ID}")
    spec = load_provider_registry().resolve(request.provider.id, request.provider.expected_config_hash)
    source = args.contactgen_root.resolve()
    verify_contactgen_source(source)
    input_path = _input_path(request, project)
    object_input = ContactGenObjectInput.model_validate(json.loads(input_path.read_text(encoding="utf-8")))
    if object_input.object.id != request.object.id or object_input.object.geometry_content_hash != request.object.geometry_content_hash:
        raise ValueError("ContactGen object input does not match request object/geometry")
    mesh_path = (project / object_input.mesh_path).resolve()
    if not mesh_path.is_relative_to(project):
        raise ValueError("ContactGen mesh path escapes project root")
    mesh_path = _file(mesh_path, "object mesh")
    if file_sha256(mesh_path) != object_input.mesh_sha256:
        raise ValueError("ContactGen input mesh hash does not match declared mesh_sha256")
    checkpoint = verify_runtime_asset(args.checkpoint, args.checkpoint_sha256, "checkpoint")
    hand_checkpoint = verify_runtime_asset(args.hand_sdf_checkpoint or source / "contactgen/hand_sdf/hand_model.pt", args.hand_sdf_checkpoint_sha256, "hand-SDF checkpoint")
    mano_root = (args.mano_root or source / "mano/models").resolve()
    mano_asset = verify_runtime_asset(mano_root / "MANO_RIGHT.pkl", args.mano_asset_sha256, "right MANO asset")
    if request.seed + request.sample_count - 1 > 2**32 - 1:
        raise ValueError("ContactGen sample seed exceeds unsigned 32-bit range")
    output = args.output.resolve()
    if any((output / f"sample_{i:04d}").exists() for i in range(request.sample_count)):
        raise FileExistsError("ContactGen worker output already has sample directories")

    mesh, mesh_center = validate_contactgen_mesh(object_input, project)

    # Heavy CUDA and licensed-asset imports are deliberately isolated here.
    import numpy as np
    import torch
    import trimesh
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("ContactGen fresh inference requires its external CUDA runtime")
    sys.path.insert(0, str(source))
    model_module = _official_module("contactgen.model", source)
    cfg_module = _official_module("contactgen.utils.cfg_parser", source)
    mano_module = _official_module("manopth.manolayer", source)
    hand_module = _official_module("contactgen.hand_sdf.hand_model", source)
    optimizer_module = _official_module("contactgen.contact.contact_optimizer", source)
    device = torch.device("cuda")
    config = cfg_module.Config(default_cfg_path=str(source / "contactgen/configs/default.yaml"))
    model = model_module.ContactGenModel(config).to(device)
    # These are explicitly supplied, trusted local runtime checkpoints.
    model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"], strict=True)
    model.eval()
    hand_config = OmegaConf.load(source / "contactgen/hand_sdf/config.yaml")
    hand_model = hand_module.ArtiHand(hand_config["model_params"], pose_size=hand_config["pose_size"])
    hand_model.load_state_dict(torch.load(hand_checkpoint, map_location=device)["state_dict"], strict=True)
    hand_model.eval().to(device)
    mano_layer = mano_module.ManoLayer(
        mano_root=str(mano_root), use_pca=True, ncomps=26,
        side="right", flat_hand_mean=False,
    ).to(device)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    # This is exactly demo.py's arithmetic vertex-mean centering operation.
    centered_mesh = trimesh.Trimesh(vertices=mesh_vertices - mesh_center, faces=mesh.faces)
    samples = []
    for sample_index in range(request.sample_count):
        seed = request.seed + sample_index
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Older Trimesh uses NumPy's seeded global RNG; current releases accept
        # an explicit seed. Keep the official 2048 area-weighted surface points.
        sample_kwargs = {"seed": seed} if "seed" in inspect.signature(trimesh.sample.sample_surface).parameters else {}
        surface, face_indices = trimesh.sample.sample_surface(centered_mesh, 2048, **sample_kwargs)
        normals = centered_mesh.face_normals[face_indices]
        points = torch.from_numpy(surface.astype(np.float32)).unsqueeze(0).to(device)
        features = torch.from_numpy(normals.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            contacts, partitions, directions = model.sample(points, features)
        # Preserve batch dimension for sample_count=1; demo.py's squeeze()
        # would collapse it. optimize_pose expects (B,N) contact weights.
        contacts = contacts.squeeze(-1)
        partitions = partitions.argmax(dim=-1)
        global_pose, mano_pose, mano_shape, mano_translation = optimizer_module.optimize_pose(
            hand_model, mano_layer, points, contacts, partitions, directions,
            w_contact=1e-1, w_pene=3.0, w_uv=1e-2,
            global_iter=200, pose_iter=1000,
        )
        with torch.no_grad():
            vertices, frames = mano_layer(
                torch.cat((global_pose, mano_pose), dim=1),
                th_betas=mano_shape, th_trans=mano_translation,
            )
        def array(value):
            return value.detach().cpu().numpy()[0].tolist()
        samples.append(ContactGenOptimizedSample(
            sample_index=sample_index, seed=seed,
            global_pose_axis_angle_rad=array(global_pose),
            mano_pose_pca=array(mano_pose), mano_shape=array(mano_shape),
            mano_translation_m=array(mano_translation),
            hand_vertices_centered_m=array(vertices),
            hand_frames_centered_m=array(frames),
        ))
    bundle = ContactGenNativeBundle(
        schema_version="contactgen_native_bundle_v1",
        source_revision=CONTACTGEN_SOURCE_REVISION,
        generation_stage="refined_mano",
        mano_configuration="right_pca26_nonflat_mean",
        coordinate_frame="vertex_mean_centered_object_mesh", units="meters",
        object_input=object_input, mesh_vertex_mean_m=tuple(mesh_center),
        checkpoint_sha256=file_sha256(checkpoint),
        hand_sdf_checkpoint_sha256=file_sha256(hand_checkpoint),
        mano_asset_sha256=file_sha256(mano_asset),
        global_iterations=200, pose_iterations=1000, samples=tuple(samples),
    )
    return convert_contactgen_bundle(
        bundle, request, spec, project_root=project, output_directory=output,
        generation_mode="fresh_official_inference",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contactgen-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--hand-sdf-checkpoint-sha256", required=True)
    parser.add_argument("--mano-asset-sha256", required=True)
    parser.add_argument("--hand-sdf-checkpoint", type=Path)
    parser.add_argument("--mano-root", type=Path)
    args = parser.parse_args()
    outputs = run_worker(args)
    print(json.dumps({"provider_id": CONTACTGEN_PROVIDER_ID, "proposal_count": len(outputs)}))


if __name__ == "__main__":
    main()
