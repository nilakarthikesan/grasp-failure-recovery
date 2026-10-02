"""Run pinned official HUG on a validated 224px simulation observation (CPU)."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import types

import numpy as np
import torch

from grasp_failure_prediction.integrations.hug import normalize_hug_prediction
from grasp_failure_prediction.integrations.hug_preprocessing import seeded_point_cloud, tensor_content_hash
from grasp_failure_prediction.integrations.hug_rng import seed_cpu_generation
from grasp_failure_prediction.integrations.observations import check_manifest, validate_depth_encoding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hug-root", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    root = args.observation.resolve()
    checks = check_manifest(root / "observation_manifest.json", root / "checks")
    if not checks["preflight_passed"]:
        raise ValueError(checks["errors"])
    if (root / "proposal.pkl").exists():
        raise FileExistsError("Proposal already exists; use a fresh observation directory")
    package = types.ModuleType("hug")
    package.__path__ = [str(args.hug_root.resolve() / "src")]
    sys.modules["hug"] = package
    from hug.prepare_inputs import prepare_pkl
    from hug.dataloader.grasp_dataset import GraspDataset
    from hug.utils import pcl_utils
    from hug.inference import load_model, load_raw_checkpoint
    from hug.models.mano import mano_params_to_grasp_dict
    from PIL import Image
    manifest = json.loads((root / "observation_manifest.json").read_text())
    rgb = np.asarray(Image.open(root / manifest["rgb_path"]))
    if rgb.shape != (224, 224, 3):
        raise ValueError("Pilot requires 224px square inputs to preserve selection coordinates")
    depth_mm = np.asarray(Image.open(root / "depth.png"))
    depth_encoding = validate_depth_encoding(np.load(root / manifest["depth_m_path"]), depth_mm)
    K = np.load(root / manifest["intrinsics_path"])
    sample_dir = root / "inputs"
    sample = prepare_pkl(rgb, depth_mm, K, "cube", sample_dir, object_name="simulated_cube")
    with sample.open("rb") as stream:
        entry = pickle.load(stream)
    entry["object_mask"] = (root / manifest["mask_path"]).read_bytes()
    entry["condition_point"] = np.asarray(manifest["selection_uv"], dtype=np.float32)
    sample.write_bytes(pickle.dumps(entry))

    class SeededDataset(GraspDataset):
        def _build_pcl(self, depth_m, rgb_np, camera_K, point_xyz=None):
            xyz, colors = seeded_point_cloud(
                depth_m.numpy(), rgb_np, camera_K, seed=args.seed,
                backproject=pcl_utils.backproject_to_pcl, sample=pcl_utils.sample_fixed_n,
                center=point_xyz, crop_radius=self.pcl_crop_radius,
                n_points=self.n_points_input,
            )
            return torch.from_numpy(xyz), torch.from_numpy(colors)

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    dataset = SeededDataset(str(sample_dir), split="eval")
    batch = dataset[0]
    print("Validated simulated cube prepared; loading HUG on CPU", flush=True)
    start = time.perf_counter()
    model = load_model(args.checkpoint, use_ema=True, device="cpu").float().eval()
    checkpoint = load_raw_checkpoint(args.checkpoint, "cpu")
    state = checkpoint.get("ema")
    if state is not None:
        state = {key.removeprefix("module."): value for key, value in state.items()
                 if not key.startswith("n_averaged")}
    else:
        state = checkpoint["model"]
    incompatible = model.load_state_dict(state, strict=False)
    # Frozen DINO parameters may be excluded from released weights; they were
    # loaded by AutoModel. Any other missing weights invalidate this test.
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys
               if not key.startswith(("image_encoder.", "mano.mano_layer."))]
    if missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    print("Model loaded; generating one actual HUG proposal", flush=True)
    seed_cpu_generation(args.seed, torch_module=torch)
    with torch.inference_mode():
        prediction = model.sample(
            batch["point_uv"][None], batch["camera_K"][None], steps=args.steps,
            rgb=batch["rgb"][None], pcl_xyz=batch["pcl_xyz"][None],
            pcl_rgb=batch["pcl_rgb"][None],
        )[0]
        grasp = mano_params_to_grasp_dict(
            prediction, model.fixed_betas.squeeze(0), model.mano, K, model.mesh_faces,
        )
    normalize_hug_prediction(grasp)
    proposal = root / "proposal.pkl"
    proposal.write_bytes(pickle.dumps(grasp))
    report = {
        "real_hug_inference_passed": True, "device": "cpu", "dtype": "float32",
        "seed": args.seed, "sampling_steps": args.steps,
        "generation_rng": "PyTorch and libc srand reset immediately before CPU model.sample; batch size one",
        "elapsed_load_and_inference_s": time.perf_counter() - start,
        "proposal_sha256": hashlib.sha256(proposal.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "observation_hashes": checks["file_sha256"],
        "hug_depth_png_sha256": hashlib.sha256((root / "depth.png").read_bytes()).hexdigest(),
        "prepared_sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "depth_encoding": depth_encoding,
        "input_tensor_hash": tensor_content_hash({k: v for k, v in batch.items() if torch.is_tensor(v)}),
        "checkpoint_missing_keys": list(incompatible.missing_keys),
        "simulation_execution_tested": False,
    }
    (root / "inference_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
