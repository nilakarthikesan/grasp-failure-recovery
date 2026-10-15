"""Prepare frozen reserve geometry checks; capture only after a committed freeze.

No HUG inference, retargeting, grasp execution, scoring or training entry point is
called. Preparation uses the existing rigid-object loader/importer and static
compilation. The run command calls existing, unmodified capture_scene gates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import traceback
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DATASET = "kevinywu/1m-hugs"
REVISION = "56ea3f100b1a4ec38bb35e986c0dd14d38977283"
OBJECTS = ("bell_pepper", "bear_figure")
CAMERAS = (("front_right", (.25, -.35, .35)), ("front_left", (-.25, -.35, .35)))
INITIAL_GAP_M = .005
INERTIA_TOLERANCE_KG_M2 = 1e-12  # Existing source-tensor import regression tolerance.
PRIMARY_METADATA_URL = f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}"
PRIMARY_METADATA = {"id": DATASET, "sha": REVISION,
                    "cardData": {"license": "cc-by-4.0", "task_categories": ["robotics"],
                                 "tags": ["grasping", "hug"], "size_categories": ["1M<n<10M"]}}
SOURCE_PATHS = (
    "scripts/qualify_hug_reserve_assets.py", "scripts/fetch_hug_object_assets.py",
    "src/grasp_failure_prediction/integrations/sim_observations.py",
    "src/grasp_failure_prediction/integrations/observations.py",
    "src/grasp_failure_prediction/integrations/hug.py",
    "src/grasp_failure_prediction/integrations/hug_frames.py",
    "src/grasp_failure_prediction/evaluation/object_assets.py",
    "src/grasp_failure_prediction/evaluation/case_runner.py",
    "src/grasp_failure_prediction/evaluation/runner.py",
    "src/grasp_failure_prediction/evaluation/pose_validation.py",
    "src/grasp_failure_prediction/evaluation/retargeting.py",
    "src/grasp_failure_prediction/evaluation/registry.py",
    "src/grasp_failure_prediction/evaluation/schema.py",
    "src/grasp_failure_prediction/evaluation/scene_contracts.py",
    "src/grasp_failure_prediction/evaluation/recording.py",
    "src/grasp_failure_prediction/evaluation/artifacts.py",
    "src/grasp_failure_prediction/evaluation/scoring.py",
    "src/grasp_failure_prediction/evaluation/configs/environments.yaml",
    "src/grasp_failure_prediction/evaluation/configs/execution_protocols.yaml",
)


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bootstrap(runtime_project):
    # NumPy/MuJoCo are already top-loaded; bundled dependency paths cannot shadow
    # their base installation. The script's canonical source always takes priority.
    sys.path.insert(0, str(PROJECT / "src"))
    for relative in ("runs/hug_integration_check/deps",
                     "runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages"):
        path = Path(runtime_project) / relative
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))


@contextmanager
def no_physics():
    originals = {name: getattr(mujoco, name) for name in ("mj_step", "mj_step1", "mj_step2")}
    def forbidden(*args, **kwargs):
        raise RuntimeError("reserve asset preparation forbids physics integration")
    for name in originals:
        setattr(mujoco, name, forbidden)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(mujoco, name, original)


def verify_download(asset_dir, object_name):
    """Check every downloaded byte, including metadata outside the MJCF closure."""
    root = Path(asset_dir).resolve()
    manifest_path = root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("dataset") != DATASET or manifest.get("revision") != REVISION
            or manifest.get("license") != PRIMARY_METADATA["cardData"]["license"]
            or manifest.get("usage") != "geometry only; no grasp labels or trajectory annotations"):
        raise ValueError("download provenance differs from pinned geometry-only dataset")
    names = set()
    for row in manifest["files"]:
        relative = PurePosixPath(row["local_path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in names:
            raise ValueError("invalid or repeated download-relative path")
        names.add(str(relative))
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("download dependency escapes object directory")
        expected_dataset_path = f"val/medium_5/{object_name}/sim_assets/{relative}"
        if (row["dataset_path"] != expected_dataset_path
                or row["url"] != f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{expected_dataset_path}"):
            raise ValueError("download path is not this pinned reserve geometry")
        if (path.stat().st_size != row["size"] or digest(path) != "sha256:" + row["sha256"]):
            raise ValueError("download content differs from pinned manifest")
        if str(relative) not in {"object.xml", "object.urdf", "properties.json", "visual.obj"} and not (
                len(relative.parts) == 2 and relative.parts[0] == "collision" and relative.suffix == ".obj"):
            raise ValueError("reserve download contains an undeclared non-geometry file")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != names | {"download_manifest.json"}:
        raise ValueError("reserve asset directory contains unbound or missing files")
    return manifest


def _tensor(model, body):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[body])
    return np.einsum("ik,k,jk->ij", rotation.reshape(3, 3), model.body_inertia[body],
                     rotation.reshape(3, 3), optimize=False)


def static_asset_report(asset_dir, object_name):
    from grasp_failure_prediction.evaluation.object_assets import load_object_geometry, append_object_geometry
    root = Path(asset_dir).resolve()
    manifest = verify_download(root, object_name)
    geometry = load_object_geometry(root / "object.xml")
    xml = ET.fromstring(geometry.source_xml)
    body = xml.find("worldbody/body")
    # This finite qualification chooses exactly the source's identity orientation.
    # A changed source frame requires a new declaration, not an automatic pose repair.
    if (body is None or body.get("pos", "0 0 0") != "0 0 0"
            or any(key in body.attrib for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis"))):
        raise ValueError("reserve v1 requires the declared native identity body frame")
    inertial = body.find("inertial")
    values = np.array([float(x) for x in inertial.attrib["fullinertia"].split()])
    xx, yy, zz, xy, xz, yz = values
    intended = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    source_mass = float(inertial.attrib["mass"])
    eigenvalues = np.linalg.eigvalsh(intended)
    if np.any(eigenvalues <= 0) or eigenvalues[-1] > eigenvalues[:2].sum() + INERTIA_TOLERANCE_KG_M2:
        raise ValueError("native inertia is not positive and physically realizable")
    scene = ET.Element("mujoco", model="reserve_static_import")
    ET.SubElement(scene, "compiler", angle="radian")
    append_object_geometry(scene, ET.SubElement(scene, "worldbody"), geometry)
    imported = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
    source = mujoco.MjModel.from_xml_path(str(geometry.mjcf_path))
    residual = float(np.max(np.abs(_tensor(imported, 1) - intended)))
    if residual > INERTIA_TOLERANCE_KG_M2 or float(imported.body_mass[1]) != source_mass:
        raise ValueError("source-accurate imported native inertia/mass differs")
    source_com = np.array([float(x) for x in inertial.get("pos", "0 0 0").split()])
    if not np.allclose(imported.body_ipos[1], source_com, atol=1e-12, rtol=0):
        raise ValueError("native center of mass changed during import")
    properties = json.loads((root / "properties.json").read_text())
    if (properties["object_name"] != object_name or properties["mass_kg"] != source_mass
            or properties["n_collision_parts"] != len(geometry.collision_geom_indices)):
        raise ValueError("native properties metadata conflicts with compiled asset")
    for name, expected in geometry.files_sha256.items():
        downloaded = next((row for row in manifest["files"] if row["local_path"] == name), None)
        if downloaded is None or expected != "sha256:" + downloaded["sha256"]:
            raise ValueError("MJCF closure is not covered by download manifest")
    bounds = geometry.local_collision_bounds_m
    return dict(object_id="hug_" + object_name, status="PASS_STATIC_ONLY", native_mass_kg=source_mass,
                geometry_content_hash=geometry.content_hash, geometry_files_sha256=geometry.files_sha256,
                download_manifest_sha256=digest(root / "download_manifest.json"),
                downloaded_files=len(manifest["files"]), collision_geoms=len(geometry.collision_geom_indices),
                local_collision_bounds_m=bounds, initial_position_m=[0., 0., -bounds[0][2] + INITIAL_GAP_M],
                initial_orientation_xyzw=[0., 0., 0., 1.], native_inertia_tensor_kg_m2=intended.tolist(),
                source_compiler_inertia_residual_kg_m2=float(np.max(np.abs(_tensor(source, 1) - intended))),
                imported_inertia_residual_kg_m2=residual, imported_inertia_tolerance_kg_m2=INERTIA_TOLERANCE_KG_M2,
                properties_inertia_rounding_residual_kg_m2=float(np.max(np.abs(np.array(properties["inertia_local"]) - intended))),
                stationary_capture="NOT_RUN", grasp_inference="NOT_RUN", grasp_outcome=None)


def _case(record, object_name, asset_root, runtime_project, resolved, seed):
    from grasp_failure_prediction.evaluation.schema import EvaluationCase
    path = (asset_root / object_name / "object.xml").resolve().relative_to(runtime_project)
    return EvaluationCase.model_validate(dict(
        schema_version="eval_case_v1", case_id="reserve_" + object_name + "_native_v1",
        environment={"id": "adroit_shadow_tabletop_v2", "expected_config_hash": resolved[0]},
        embodiment={"id": "shadow_hand_right"},
        execution_protocol={"id": "fixed_grasp_lift_v2", "expected_config_hash": resolved[1]},
        object={"id": record["object_id"], "mass_kg": record["native_mass_kg"], "friction_profile_id": "nominal_v1",
                "geometry": {"mjcf_path": str(path), "expected_content_hash": record["geometry_content_hash"]}},
        grasp={"source": "hug", "id": "reserve_no_grasp_v1"},
        initial_condition={"id": "native_orientation_gap_5mm_v1", "object_position_m": record["initial_position_m"],
                           "object_orientation_xyzw": record["initial_orientation_xyzw"]},
        motion_profile_id="nominal_lift_v1", seed=seed)).model_dump(mode="json")


def runtime_identity():
    return {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__}


def prepare(asset_root, runtime_project, hands_root, evidence_dir):
    from grasp_failure_prediction.evaluation.registry import load_environment_registry, load_protocol_registry
    from grasp_failure_prediction.evaluation.schema import canonical_sha256
    from grasp_failure_prediction.integrations import sim_observations as capture
    asset_root, runtime_project, hands_root, evidence = map(lambda p: Path(p).resolve(),
                                                           (asset_root, runtime_project, hands_root, evidence_dir))
    if evidence.exists():
        raise FileExistsError("reserve preparation requires a fresh evidence directory")
    env = load_environment_registry().resolve("adroit_shadow_tabletop_v2")
    execution = load_protocol_registry().resolve("fixed_grasp_lift_v2")
    resolved = (canonical_sha256(env), canonical_sha256(execution))
    with no_physics():
        reports = [static_asset_report(asset_root / name, name) for name in OBJECTS]
        cases = [_case(row, name, asset_root, runtime_project, resolved, 61000 + i)
                 for i, (row, name) in enumerate(zip(reports, OBJECTS))]
    hand_dir = hands_root / "shadow_hand"
    if not (hand_dir / "shadow_hand_right.urdf").is_file():
        raise FileNotFoundError("official Shadow hand assets required")
    hand_hashes = {str(path.relative_to(hands_root)): digest(path) for path in sorted(hand_dir.rglob("*")) if path.is_file()}
    evidence.mkdir(parents=True)
    for name, case in zip(OBJECTS, cases):
        write_json(evidence / f"case_{name}.json", case)
    write_json(evidence / "primary_dataset_metadata.json", {"url": PRIMARY_METADATA_URL,
                "retrieved_utc_date": "2026-10-15", "response_fields": PRIMARY_METADATA})
    protocol = dict(schema_version="hug_reserve_asset_qualification_v1", protocol_id="hug_reserve_assets_native_v1",
        purpose="stationary geometry/camera qualification only; reserve for frozen-model evaluation",
        project_root=str(PROJECT), runtime_project=str(runtime_project), asset_root=str(asset_root), hands_root=str(hands_root),
        sources_sha256={path: digest(PROJECT / path) for path in SOURCE_PATHS},
        hand_files_sha256=hand_hashes, runtime=runtime_identity(), objects=reports,
        cases={f"case_{name}.json": digest(evidence / f"case_{name}.json") for name in OBJECTS},
        primary_metadata_sha256=digest(evidence / "primary_dataset_metadata.json"),
        license={"dataset": DATASET, "revision": REVISION, "license": PRIMARY_METADATA["cardData"]["license"],
                 "metadata_url": PRIMARY_METADATA_URL, "attribution": "1M-HuGs / HUG authors; pinned original geometry paths in download manifests"},
        capture_schedule=[dict(capture_id=f"{name}_{camera}", object_id="hug_"+name,
                              case_file=f"case_{name}.json", camera_id=camera, camera_position_m=list(position))
                          for name in OBJECTS for camera, position in CAMERAS],
        gates=dict(window_samples=capture.SETTLING_WINDOW_SAMPLES, max_physics_steps_per_capture=capture.SETTLING_MAX_PHYSICS_STEPS,
                   max_position_diagonal_m=capture.SETTLING_MAX_POSITION_DIAGONAL_M,
                   max_orientation_diameter_rad=capture.SETTLING_MAX_ORIENTATION_DIAMETER_RAD,
                   workspace_half_width_m=capture.SETTLING_WORKSPACE_HALF_WIDTH_M,
                   continuous_object_table_contact=True, no_hand_object_contact=True,
                   maximum_ray_surface_error_m=.002, velocity_policy="finite; no additional speed thresholds",
                   captured_image_size=[224,224], fovy_degrees=45.),
        max_physics_steps=4*capture.SETTLING_MAX_PHYSICS_STEPS, initial_gap_m=INITIAL_GAP_M,
        no_automatic_replacement=True, grasp_inference_allowed=False, grasp_labels_allowed=False,
        fitting_allowed=False, reserve_use="fresh frozen-model evaluation only after model freeze",
        preparation_physics_steps=0, stationary_capture_status="NOT_RUN")
    write_json(evidence / "protocol.json", protocol)
    return protocol


def verify_protocol(protocol_path, freeze_commit):
    path = Path(protocol_path).resolve()
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != "hug_reserve_asset_qualification_v1":
        raise ValueError("unknown reserve protocol")
    if protocol["runtime"] != runtime_identity():
        raise ValueError("runtime differs from frozen reserve protocol")
    # A concrete commit must contain the exact protocol, cases and implementation.
    bindings = {**protocol["sources_sha256"], str(path.relative_to(PROJECT)): digest(path),
                **{str((path.parent/name).relative_to(PROJECT)): expected for name, expected in protocol["cases"].items()},
                str((path.parent/"primary_dataset_metadata.json").relative_to(PROJECT)): protocol["primary_metadata_sha256"]}
    for relative, expected in bindings.items():
        if digest(PROJECT / relative) != expected:
            raise ValueError(f"bound source/input changed: {relative}")
        blob = subprocess.run(["git", "show", f"{freeze_commit}:{relative}"], cwd=PROJECT, check=True, capture_output=True).stdout
        if "sha256:" + hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"freeze commit does not contain exact source/input: {relative}")
    for relative, expected in protocol["hand_files_sha256"].items():
        if digest(Path(protocol["hands_root"]) / relative) != expected:
            raise ValueError("bound Shadow asset changed")
    with no_physics():
        reports = [static_asset_report(Path(protocol["asset_root"])/name, name) for name in OBJECTS]
    if canonical(reports) != canonical(protocol["objects"]):
        raise ValueError("reserve static checks differ from frozen report")
    return protocol


def run(protocol_path, output_dir, *, freeze_commit):
    protocol = verify_protocol(protocol_path, freeze_commit)
    from grasp_failure_prediction.integrations.sim_observations import capture_scene
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("stationary capture output must be a fresh directory")
    output.mkdir(parents=True)
    report = dict(protocol_sha256=digest(protocol_path), freeze_commit=freeze_commit,
                  scope="stationary reserve asset/camera qualification; no grasps or task labels", captures=[],
                  grasp_inference_calls=0, grasp_labels=0, model_fits=0)
    write_json(output / "capture_qualification.json", report)
    for row in protocol["capture_schedule"]:
        item = dict(row, status="NOT_RUN", outcome=None)
        report["captures"].append(item)
        try:
            result = capture_scene(Path(protocol_path).parent / row["case_file"], output / row["capture_id"],
                                   project_root=protocol["runtime_project"], urdf_root=protocol["hands_root"],
                                   camera_position_m=tuple(row["camera_position_m"]))
            item.update(status="PASS_STATIONARY_CAPTURE", capture_report=result,
                        capture_report_sha256=digest(output/row["capture_id"]/"capture_report.json"))
        except Exception as exc:
            item.update(status="CAPTURE_REJECTED", error_type=type(exc).__name__, error=str(exc))
            (output / (row["capture_id"] + "_rejection.txt")).write_text(traceback.format_exc())
        write_json(output / "capture_qualification.json", report)
    report["verdict"] = "PASS_STATIONARY_ONLY" if all(row["status"] == "PASS_STATIONARY_CAPTURE" for row in report["captures"]) else "INCOMPLETE"
    write_json(output / "capture_qualification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-project", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("asset-root", "hands-root", "evidence-dir"):
        prep.add_argument("--"+name, type=Path, required=True)
    capture = commands.add_parser("run")
    capture.add_argument("--protocol", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    bootstrap(args.runtime_project)
    if args.command == "prepare":
        result = prepare(args.asset_root, args.runtime_project, args.hands_root, args.evidence_dir)
        print(json.dumps({"protocol_sha256": digest(args.evidence_dir/"protocol.json"), "captures": len(result["capture_schedule"]), "physics_steps": 0}))
    else:
        result = run(args.protocol, args.output, freeze_commit=args.freeze_commit)
        print(json.dumps({"verdict": result["verdict"], "captures": len(result["captures"])}))


if __name__ == "__main__":
    main()
