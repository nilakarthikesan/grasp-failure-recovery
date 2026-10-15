"""Freeze a controller, then generate and execute fresh immutable HUG proposals.

This pilot varies generation seeds on one observation. It is neither an official
HUG-Bench reproduction nor a held-out object/physics experiment. No object-aware
grasp optimizer is called. The input point-cloud subset is fixed across seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import types

import mujoco
import numpy as np
from PIL import Image
import torch

from actuated_shadow_pilot import ActuatedShadowRunner, run_candidate
from check_contact_execution import setup
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.integrations.hug import normalize_hug_prediction
from grasp_failure_prediction.integrations.hug_preprocessing import seeded_point_cloud, tensor_content_hash
from grasp_failure_prediction.integrations.hug_rng import seed_cpu_generation
from grasp_failure_prediction.integrations.observations import check_manifest, validate_depth_encoding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--hug-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--urdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--point-cloud-seed", type=int, default=42)
    parser.add_argument("--force-close-delta-rad", type=float, default=.1)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Generation seeds must be unique and nonnegative")
    root = args.observation.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    checks = check_manifest(root / "observation_manifest.json", args.output / "checks")
    if not checks["preflight_passed"]:
        raise ValueError(checks["errors"])
    png = np.asarray(Image.open(root / "depth.png"))
    validate_depth_encoding(np.load(root / "depth_m.npy"), png)
    package = types.ModuleType("hug")
    package.__path__ = [str(args.hug_root.resolve() / "src")]
    sys.modules["hug"] = package
    from hug.dataloader.grasp_dataset import GraspDataset
    from hug.inference import load_model, load_raw_checkpoint
    from hug.models.mano import mano_params_to_grasp_dict
    from hug.prepare_inputs import prepare_pkl
    from hug.utils import pcl_utils

    # Use the already prepared official input, checking it against source files.
    prepared = pickle.loads((root / "inputs/cube.pkl").read_bytes())
    np.testing.assert_array_equal(GraspDataset._decode_depth_uint16(prepared["depth"]), png)
    K = np.load(root / "intrinsics.npy")
    np.testing.assert_array_equal(prepared["camera"]["K"], K)
    # Upstream stores RGB as JPEG. Compare to its encoding of the source PNG,
    # rather than incorrectly requiring lossy JPEG pixels to equal PNG pixels.
    with tempfile.TemporaryDirectory() as temporary:
        expected_path = prepare_pkl(np.asarray(Image.open(root / "rgb.png")),
                                    png, K, "source_check", Path(temporary))
        expected = pickle.loads(expected_path.read_bytes())
    if prepared["image"] != expected["image"]:
        raise ValueError("Prepared RGB differs from official encoding of the source capture")
    manifest = json.loads((root / "observation_manifest.json").read_text())
    np.testing.assert_array_equal(prepared["condition_point"], manifest["selection_uv"])

    class FixedPointCloudDataset(GraspDataset):
        def _build_pcl(self, depth_m, rgb_np, camera_K, point_xyz=None):
            xyz, colors = seeded_point_cloud(
                depth_m.numpy(), rgb_np, camera_K, seed=args.point_cloud_seed,
                backproject=pcl_utils.backproject_to_pcl, sample=pcl_utils.sample_fixed_n,
                center=point_xyz, crop_radius=self.pcl_crop_radius, n_points=self.n_points_input,
            )
            return torch.from_numpy(xyz), torch.from_numpy(colors)

    torch.set_num_threads(4)
    batch = FixedPointCloudDataset(str(root / "inputs"), split="eval")[0]
    model = load_model(args.checkpoint, use_ema=True, device="cpu").float().eval()
    checkpoint = load_raw_checkpoint(args.checkpoint, "cpu")
    # Inspect the selected state without replacing EMA weights with raw weights.
    state = checkpoint.get("ema")
    if state is not None:
        state = {key.removeprefix("module."): value for key, value in state.items()
                 if not key.startswith("n_averaged")}
    else:
        state = checkpoint["model"]
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys
               if not key.startswith(("image_encoder.", "mano.mano_layer."))]
    if missing or incompatible.unexpected_keys:
        raise ValueError("Checkpoint has missing or unexpected learned parameters")
    protocol = load_protocol_registry().resolve("fixed_grasp_lift_v1")
    runner = ActuatedShadowRunner(protocol, args.urdf_root,
                                  force_close_delta_rad=args.force_close_delta_rad)
    comparison_runner = ActuatedShadowRunner(protocol, args.urdf_root)
    snapshot = np.load(root / "scene_state.npz")
    report = {
        "device": "cpu", "dtype": "float32", "sampling_steps": 50,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "mujoco": mujoco.__version__},
        "controller_source_sha256": hashlib.sha256(
            Path(__file__).with_name("actuated_shadow_pilot.py").read_bytes()).hexdigest(),
        "rng_reset": "PyTorch and libc srand immediately before each CPU model.sample; batch size one",
        "point_cloud_seed": args.point_cloud_seed,
        "input_tensor_hash": tensor_content_hash({k: v for k, v in batch.items() if torch.is_tensor(v)}),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "observation_hashes": checks["file_sha256"],
        "source_rgb_verified_after_official_jpeg_preparation": True,
        "force_close_delta_rad": args.force_close_delta_rad,
        "object_aware_contact_fit": False,
        "scope": "Fresh seeds on the same pilot observation; not independent object/physics holdouts",
        "trials": [],
    }
    first_prediction = None
    for seed in args.seeds:
        directory = args.output / f"seed_{seed}"
        directory.mkdir(exist_ok=False)
        for name in ("T_world_camera.npy", "scene_state.npz"):
            shutil.copyfile(root / name, directory / name)
        seed_cpu_generation(seed, torch_module=torch)
        with torch.inference_mode():
            prediction = model.sample(batch["point_uv"][None], batch["camera_K"][None],
                                      steps=50, rgb=batch["rgb"][None],
                                      pcl_xyz=batch["pcl_xyz"][None], pcl_rgb=batch["pcl_rgb"][None])[0]
            grasp = mano_params_to_grasp_dict(prediction, model.fixed_betas.squeeze(0),
                                              model.mano, K, model.mesh_faces)
        if first_prediction is None:
            first_prediction = prediction.detach().clone()
        normalize_hug_prediction(grasp)
        proposal = directory / "proposal.pkl"
        proposal.write_bytes(pickle.dumps(grasp))
        before_hash = hashlib.sha256(proposal.read_bytes()).hexdigest()
        pose, position, quaternion = setup(directory, args.urdf_root)
        result, trace = run_candidate(runner, pose, position, quaternion, snapshot)
        comparison, comparison_trace = run_candidate(comparison_runner, pose, position, quaternion, snapshot)
        if hashlib.sha256(proposal.read_bytes()).hexdigest() != before_hash:
            raise ValueError("Immutable proposal changed during execution")
        result.update({"generation_seed": seed, "proposal_sha256": before_hash,
                       "proposal_modified": False, "object_aware_contact_fit": False,
                       "no_force_close_comparison": comparison})
        np.savez(directory / "trace.npz", qpos=np.stack(runner.saved_qpos),
                 time_s=[s.time_s for s in trace.steps], phase=[s.state.value for s in trace.steps])
        np.savez(directory / "no_closure_trace.npz", qpos=np.stack(comparison_runner.saved_qpos),
                 time_s=[s.time_s for s in comparison_trace.steps],
                 phase=[s.state.value for s in comparison_trace.steps])
        (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        report["trials"].append(result)
        print(json.dumps({"seed": seed, "score": result["score"],
                          "no_closure_success": comparison["score"]["success"]}), flush=True)
    seed_cpu_generation(args.seeds[0], torch_module=torch)
    with torch.inference_mode():
        repeated = model.sample(batch["point_uv"][None], batch["camera_K"][None],
                                steps=50, rgb=batch["rgb"][None], pcl_xyz=batch["pcl_xyz"][None],
                                pcl_rgb=batch["pcl_rgb"][None])[0]
    report["first_seed_parameters_repeat_exactly"] = torch.equal(first_prediction, repeated)
    if not report["first_seed_parameters_repeat_exactly"]:
        raise ValueError("Repeated first generation seed produced different parameters")
    mujoco.mj_saveLastXML(str(args.output / "scene.xml"), runner.model)
    report["successes"] = sum(trial["score"]["success"] for trial in report["trials"])
    report["no_force_close_successes"] = sum(
        trial["no_force_close_comparison"]["score"]["success"] for trial in report["trials"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
