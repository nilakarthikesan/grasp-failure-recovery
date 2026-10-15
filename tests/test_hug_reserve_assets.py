"""Manufactured rigid assets and mocked capture calls; zero real integration."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/qualify_hug_reserve_assets.py"
spec = importlib.util.spec_from_file_location("reserve_asset_qualification", PATH)
reserve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reserve)


@pytest.fixture(autouse=True)
def forbid_physics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("reserve qualification development tests forbid real integration")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mujoco, name, forbidden)


def write_asset(root, name):
    root.mkdir(parents=True)
    (root / "collision").mkdir()
    vertices = [(x*.02, y*.03, z*.04) for x,y,z in
                ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))]
    faces = ((1,3,2),(1,4,3),(5,6,7),(5,7,8),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,4,8),(3,8,7),(4,1,5),(4,5,8))
    mesh = "\n".join("v "+" ".join(map(str,row)) for row in vertices)+"\n"
    mesh += "\n".join("f "+" ".join(map(str,row)) for row in faces)+"\n"
    (root / "visual.obj").write_text(mesh)
    (root / "collision/part_000.obj").write_text(mesh)
    (root / "object.xml").write_text(f'''<mujoco model="{name}"><compiler meshdir="." angle="radian"/>
      <asset><mesh name="visual" file="visual.obj"/><mesh name="collision" file="collision/part_000.obj"/></asset>
      <worldbody><body name="{name}" pos="0 0 0"><inertial pos="0 0 0" mass=".2"
      fullinertia=".00025 .00035 .00030 .00001 -.00002 .000005"/><freejoint/>
      <geom type="mesh" mesh="visual" contype="0" conaffinity="0" group="2"/>
      <geom type="mesh" mesh="collision" friction="1 .005 .0001" group="3"/>
      </body></worldbody></mujoco>''')
    (root / "object.urdf").write_text('<robot name="unused_metadata"/>')
    (root / "properties.json").write_text(json.dumps(dict(object_name=name,mass_kg=.2,n_collision_parts=1,
                inertia_local=[[.00025,.00001,-.00002],[.00001,.00035,.000005],[-.00002,.000005,.00030]])))
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        dataset_path = f"val/medium_5/{name}/sim_assets/{relative}"
        rows.append(dict(local_path=relative,dataset_path=dataset_path,size=path.stat().st_size,
                         sha256=reserve.digest(path)[7:],
                         url=f"https://huggingface.co/datasets/{reserve.DATASET}/resolve/{reserve.REVISION}/{dataset_path}"))
    reserve.write_json(root/"download_manifest.json",dict(dataset=reserve.DATASET,revision=reserve.REVISION,
        license="cc-by-4.0",usage="geometry only; no grasp labels or trajectory annotations",files=rows))


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "runtime/runs/reserve_assets"
    for name in reserve.OBJECTS:
        write_asset(root/name,name)
    return root


def test_static_native_import_and_collision_height_have_no_task_claim(assets):
    report = reserve.static_asset_report(assets/"bell_pepper","bell_pepper")
    assert report["native_mass_kg"] == .2 and report["collision_geoms"] == 1
    assert report["initial_position_m"][2] == pytest.approx(.045, abs=1e-8)
    assert report["initial_orientation_xyzw"] == [0.,0.,0.,1.]
    assert report["imported_inertia_residual_kg_m2"] <= 1e-12
    assert report["stationary_capture"] == "NOT_RUN" and report["grasp_outcome"] is None


@pytest.mark.parametrize("change", ["mesh", "unbound_file", "license", "dataset_path", "escape", "duplicate"])
def test_asset_provenance_or_closure_changes_refused(assets, change):
    root = assets/"bell_pepper"
    manifest = json.loads((root/"download_manifest.json").read_text())
    if change == "mesh":
        (root/"visual.obj").write_text("changed")
    elif change == "unbound_file":
        (root/"grasps.json").write_text("{}")
    elif change == "license":
        manifest["license"] = "unknown"
    elif change == "dataset_path":
        manifest["files"][0]["dataset_path"] = "val/foreign/labels.json"
    elif change == "escape":
        manifest["files"][0]["local_path"] = "../outside.obj"
    else:
        manifest["files"].append(manifest["files"][0])
    reserve.write_json(root/"download_manifest.json",manifest)
    with pytest.raises(ValueError):
        reserve.static_asset_report(root,"bell_pepper")


def test_prepare_freezes_two_cases_four_captures_and_existing_gates(assets,tmp_path):
    hands = tmp_path/"hands/shadow_hand"
    hands.mkdir(parents=True)
    (hands/"shadow_hand_right.urdf").write_text("manufactured identity only; no capture")
    evidence = tmp_path/"prepared"
    protocol = reserve.prepare(assets,assets.parents[1],hands.parent,evidence)
    assert len(protocol["objects"]) == 2 and len(protocol["capture_schedule"]) == 4
    assert protocol["object_names"] == list(reserve.OBJECTS)
    assert protocol["max_physics_steps"] == 8000
    assert protocol["preparation_physics_steps"] == 0
    assert protocol["gates"]["window_samples"] == 500
    assert protocol["gates"]["max_position_diagonal_m"] == .0001
    assert protocol["gates"]["max_orientation_diameter_rad"] == .002
    assert not protocol["grasp_labels_allowed"] and not protocol["fitting_allowed"]
    for name in reserve.OBJECTS:
        case = json.loads((evidence/f"case_{name}.json").read_text())
        assert case["object"]["id"] == "hug_"+name
        assert case["object"]["mass_kg"] == .2
        assert case["grasp"]["prediction_path"] is None
    with pytest.raises(FileExistsError):
        reserve.prepare(assets,assets.parents[1],hands.parent,evidence)


def test_capture_failures_retained_without_replacement_or_labels(tmp_path,monkeypatch):
    from grasp_failure_prediction.integrations import sim_observations
    protocol_path = tmp_path/"protocol.json"
    protocol_path.write_text("{}")
    schedule = [dict(capture_id=f"case_{i}",case_file="unused.json",camera_position_m=[.25,-.35,.35]) for i in range(4)]
    monkeypatch.setattr(reserve,"verify_protocol",lambda *a:dict(capture_schedule=schedule,runtime_project=str(tmp_path),hands_root="unused"))
    calls = []
    def capture(case,output,**kwargs):
        calls.append(str(output))
        if len(calls) == 2:
            raise ValueError("manufactured stationary rejection")
        output.mkdir()
        reserve.write_json(output/"capture_report.json",{"input_checks_passed":True})
        return {"input_checks_passed":True}
    monkeypatch.setattr(sim_observations,"capture_scene",capture)
    output = tmp_path/"captured"
    report = reserve.run(protocol_path,output,freeze_commit="HEAD")
    assert len(calls) == 4 and len(report["captures"]) == 4
    assert report["captures"][1]["status"] == "CAPTURE_REJECTED"
    assert "manufactured stationary rejection" in (output/"case_1_rejection.txt").read_text()
    assert all(row["outcome"] is None for row in report["captures"])
    assert report["verdict"] == "INCOMPLETE"
    assert report["grasp_inference_calls"] == report["grasp_labels"] == report["model_fits"] == 0
    assert len(report["freeze_commit"]) == 40 and report["freeze_commit"] != "HEAD"


def test_explicit_coffee_selection_has_only_its_case_captures_and_quota(assets,tmp_path):
    write_asset(assets/"coffee_cup","coffee_cup")
    hands = tmp_path/"hands/shadow_hand"
    hands.mkdir(parents=True)
    (hands/"shadow_hand_right.urdf").write_text("manufactured identity only")
    evidence = tmp_path/"coffee_prepared"
    protocol = reserve.prepare(assets,assets.parents[1],hands.parent,evidence,
                               objects=("coffee_cup",),protocol_id="hug_reserve_assets_coffee_v2")
    assert protocol["object_names"] == ["coffee_cup"]
    assert protocol["protocol_id"] == "hug_reserve_assets_coffee_v2"
    assert len(protocol["objects"]) == 1 and protocol["objects"][0]["object_id"] == "hug_coffee_cup"
    assert set(protocol["cases"]) == {"case_coffee_cup.json"}
    assert len(protocol["capture_schedule"]) == 2 and protocol["max_physics_steps"] == 4000
    assert all(row["object_id"] == "hug_coffee_cup" for row in protocol["capture_schedule"])
    assert not (evidence/"case_bear_figure.json").exists()


@pytest.mark.parametrize("names", [(),("coffee_cup","coffee_cup"),("../coffee_cup",),("/coffee_cup",),
                                  ("Coffee Cup",),(None,),tuple(f"object_{i}" for i in range(17))])
def test_object_selection_rejects_empty_duplicate_escaping_or_unbounded_names(names):
    with pytest.raises(ValueError,match="object names"):
        reserve.object_names(names)


def test_capture_cannot_start_before_frozen_protocol_verification(tmp_path,monkeypatch):
    from grasp_failure_prediction.integrations import sim_observations
    monkeypatch.setattr(reserve,"verify_protocol",lambda *a:(_ for _ in ()).throw(ValueError("not committed")))
    monkeypatch.setattr(sim_observations,"capture_scene",lambda *a,**kw:pytest.fail("freeze first"))
    with pytest.raises(ValueError,match="not committed"):
        reserve.run(tmp_path/"missing",tmp_path/"output",freeze_commit="HEAD")
    assert not (tmp_path/"output").exists()


def test_freeze_reference_must_resolve_to_exact_commit(monkeypatch):
    calls = []
    def resolve(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="a"*40+"\n")
    monkeypatch.setattr(reserve.subprocess,"run",resolve)
    assert reserve.resolve_freeze_commit("HEAD") == "a"*40
    assert calls == [["git","rev-parse","--verify","--end-of-options","HEAD^{commit}"]]
    monkeypatch.setattr(reserve.subprocess,"run",lambda *a,**kw:SimpleNamespace(stdout="HEAD\n"))
    with pytest.raises(ValueError,match="exact 40-character"):
        reserve.resolve_freeze_commit("HEAD")


def test_static_scope_blocks_all_three_integration_apis():
    with reserve.no_physics():
        for name in ("mj_step","mj_step1","mj_step2"):
            with pytest.raises(RuntimeError,match="forbids physics"):
                getattr(mujoco,name)(None,None)
