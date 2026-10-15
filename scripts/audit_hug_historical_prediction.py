#!/usr/bin/env python3
"""Versioned historical-weight and archived-LOO replay; fitting is separate.

Only the original 1040/2080 portable datasets are admitted. Audit forbids all
optimizers and MuJoCo integration. Refit requires an explicit structured,
committed declaration and a fresh output; no refit runs during preparation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import mujoco


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("object_scene_pilot_v1", "fresh_object_scene_eval_v1")
EXPECTED_ROWS = (1040, 2080)
ARCHIVE = "hug_retrospective_object_holdout_2026-10-15_v1"
MODEL_NAMES = ("training_prevalence", "proprioception_and_commands", "commands_only")
SOURCE_NAMES = ("scripts/audit_hug_historical_prediction.py", "scripts/verify_historical_prediction.py",
    "src/grasp_failure_prediction/prediction/dataset.py", "src/grasp_failure_prediction/prediction/model.py",
    "src/grasp_failure_prediction/prediction/fresh_evaluation.py",
    "src/grasp_failure_prediction/prediction/retrospective_holdout.py", "docs/HUG_HISTORICAL_PREDICTION_AUDIT_V1.md")


def sha(path):
    return "sha256:"+hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return "sha256:"+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open("x") as handle:
        json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write("\n")


def bootstrap(runtime_project):
    sys.path[:0] = [str(PROJECT/"src"),str(PROJECT/"scripts")]
    dependency = Path(runtime_project)/"runs/hug_integration_check/deps"
    if dependency.is_dir() and str(dependency) not in sys.path:
        sys.path.append(str(dependency))


@contextmanager
def guarded(*,allow_fitting=False):
    from grasp_failure_prediction.prediction import model,retrospective_holdout
    counts = dict(integration_api_attempts=0,optimization_attempts=0)
    originals = []
    def block_physics(*a,**kw):
        counts["integration_api_attempts"] += 1
        raise RuntimeError("historical prediction replay forbids physics")
    def block_fit(*a,**kw):
        counts["optimization_attempts"] += 1
        raise RuntimeError("read-only historical replay forbids fitting")
    targets = [(mujoco,name,block_physics) for name in ("mj_step","mj_step1","mj_step2")]
    if allow_fitting:
        original_fit = model.fit_logistic
        def counted_fit(*args,**kwargs):
            counts["optimization_attempts"] += 1
            return original_fit(*args,**kwargs)
        targets.append((model,"fit_logistic",counted_fit))
    else:
        targets += [(model,name,block_fit) for name in ("fit_logistic","select_model","train_baselines")]
        targets += [(retrospective_holdout,"select_model",block_fit)]
    for owner,name,replacement in targets:
        originals.append((owner,name,getattr(owner,name)));setattr(owner,name,replacement)
    try:
        yield counts
    finally:
        for owner,name,value in reversed(originals):setattr(owner,name,value)


def execution_cohort(manifests):
    """Match original executable collector/scorer identities, never new runners."""
    reference = manifests[1]["execution_reference"]
    if (reference.get("embodiment_id")!="shadow_hand_right"
            or reference["environment"]["environment_id"]!="adroit_shadow_tabletop_v2"
            or reference["execution_protocol"]["execution_protocol_id"]!="fixed_grasp_lift_v2"
            or reference["environment"]["runner"]!="grasp_failure_prediction.evaluation.runner:ActuatedShadowRunner"):
        raise ValueError("historical replay excludes changed hand/controller protocols")
    for key,hash_key in (("environment","environment_config_hash"),("execution_protocol","execution_protocol_config_hash")):
        if canonical(reference[key])!=reference[hash_key]:raise ValueError("historical execution reference hash mismatch")
    if reference["source_resolved_case_sha256"] not in {
            row["raw_files_sha256"]["resolved_case.json"].removeprefix("sha256:") for row in manifests[0]["episodes"]}:
        raise ValueError("fresh execution reference does not bind an original resolved case")
    signatures = set()
    for manifest in manifests:
        for source in manifest["provenance_sources"].values():
            files = source["source_file_sha256"]
            signatures.add(tuple(files[key] for key in ("runner.py","noisy_rollouts.py","scoring.py")))
    if len(signatures)!=1:raise ValueError("historical collector/scorer source identities cannot be pooled")
    return dict(hand_id=reference["embodiment_id"],environment_id="adroit_shadow_tabletop_v2",
                execution_protocol_id="fixed_grasp_lift_v2",environment_config_hash=reference["environment_config_hash"],
                execution_protocol_config_hash=reference["execution_protocol_config_hash"],
                collector_source_signature=list(next(iter(signatures))))


def load_historical(project):
    from grasp_failure_prediction.prediction.fresh_evaluation import _freeze
    from grasp_failure_prediction.prediction.retrospective_holdout import load_sources
    paths = [Path(project)/"research_artifacts"/name for name in DATASETS]
    original,_,_,frozen_models = _freeze(paths[0],Path(project)/"research_artifacts/task_outcome_baseline_v1")
    data,ids,bindings = load_sources(paths)
    manifests = [read(path/"manifest.json") for path in paths]
    if any(manifests[1]["frozen_artifacts"].get(key)!=value for key,value in frozen_models.items()):
        raise ValueError("fresh historical manifest differs from the immutable original models/inputs")
    for manifest,n,expected_schema in zip(manifests,EXPECTED_ROWS,("task_prefix_dataset_v1","fresh_task_prefix_dataset_v1"),strict=True):
        if (manifest["schema_version"]!=expected_schema or len(manifest["episodes"])!=n
                or manifest["input_contract"]["fields"]!=["hand_qpos","hand_qvel","commanded_target_rad"]):
            raise ValueError("only the literal historical1040/2080 prefix schemas are admitted")
    return data,ids,manifests,original,execution_cohort(manifests)


def compare_metrics(actual,expected,*,path="metrics",atol=1e-12):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError(path+" keys differ")
        for key in expected:compare_metrics(actual[key],expected[key],path=path+"."+key,atol=atol)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError(path+" length differs")
        for i,(a,b) in enumerate(zip(actual,expected,strict=True)):compare_metrics(a,b,path=f"{path}[{i}]",atol=atol)
    elif isinstance(expected,float):
        if actual is None or not np.isfinite(actual) or abs(actual-expected)>atol:raise ValueError(path+" numeric value differs")
    elif actual!=expected:
        raise ValueError(path+" value differs")


def replay_archived_loo(archive,data,ids):
    from grasp_failure_prediction.prediction.retrospective_holdout import object_folds,calibration
    from grasp_failure_prediction.prediction.model import outcome_metrics
    archive = Path(archive);report = read(archive/"report.json")
    if (report.get("schema_version")!="hug_retrospective_object_holdout_v1" or report.get("attempts")!=len(ids)
            or report.get("successes")!=int(data["y"].sum()) or report.get("failures")!=int(len(ids)-data["y"].sum())):
        raise ValueError("archived LOO schema/attempt denominator differs")
    if (sha(archive/"predictions.csv").removeprefix("sha256:")!=report["predictions_sha256"]
            or sha(archive/"source/retrospective_holdout.py").removeprefix("sha256:")!=report["source_sha256"]
            or sha(archive/"source/HUG_RETROSPECTIVE_EVALUATION_V1.md").removeprefix("sha256:")!=report["protocol_sha256"]):
        raise ValueError("archived LOO payload/source/protocol hash mismatch")
    folds = object_folds(data["geometry_group_ids"],data["observation_group_ids"],data["group_ids"])
    if len(report["folds"])!=len(folds):raise ValueError("archived LOO fold count differs")
    predictions = defaultdict(dict)
    with (archive/"predictions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["held_geometry"],row["model"]
            if row["episode_id"] in predictions[key]:raise ValueError("duplicate archived LOO prediction")
            predictions[key][row["episode_id"]] = (int(row["success"]),float(row["p_success"]))
    expected_keys = {(fold["held_geometry"],name) for fold in folds for name in MODEL_NAMES}
    if set(predictions)!=expected_keys:raise ValueError("archived LOO model/fold inventory differs")
    verified = []
    for fold,stored in zip(folds,report["folds"],strict=True):
        if fold["held_geometry"]!=stored["held_geometry"]:raise ValueError("archived LOO held geometry differs")
        for role,index in fold["indices"].items():
            part = stored["partitions"][role]
            if (part["episodes"]!=[ids[i] for i in index] or part["n"]!=len(index)
                    or part["observations"]!=len(set(data["observation_group_ids"][index]))
                    or part["proposals"]!=len(set(data["group_ids"][index]))):
                raise ValueError("archived LOO membership/denominator differs")
        test = fold["indices"]["test"];current = dict(held_geometry=fold["held_geometry"],models={})
        for name in MODEL_NAMES:
            values = predictions[fold["held_geometry"],name]
            if set(values)!={ids[i] for i in test}:raise ValueError("archived LOO predictions do not cover test membership")
            if any(values[ids[i]][0]!=int(data["y"][i]) for i in test):raise ValueError("archived LOO label differs")
            probability = np.array([values[ids[i]][1] for i in test])
            metrics = outcome_metrics(data["y"][test],probability)
            compare_metrics(metrics,stored["models"][name]["metrics"])
            compare_metrics(calibration(data["y"][test],probability),stored["models"][name]["calibration"])
            current["models"][name] = metrics
        verified.append(current)
    return dict(status="VERIFIED_PERSISTED_PREDICTIONS_ONLY",original_fit_commit=report["commit"],
                attempts=len(ids),folds=verified,optimization_calls=0,
                coefficients_available=False,coefficient_reproduction="NOT_RUN_ARCHIVE_DID_NOT_SAVE_LOO_WEIGHTS",
                uncertainty="Archived proposal-family bootstrap not rerun; proposals share captures, so intervals are conditional development diagnostics")


def input_bindings(project):
    root = Path(project)/"research_artifacts"
    paths = [root/name/file for name in DATASETS for file in ("manifest.json","dataset.npz")]
    paths += [root/DATASETS[0]/"splits.json"]
    paths += [root/"task_outcome_baseline_v1"/file for file in ("report.json","proprioception_and_commands.npz","commands_only.npz")]
    paths += [root/ARCHIVE/file for file in ("report.json","predictions.csv","historical_readback.json",
               "source/retrospective_holdout.py","source/HUG_RETROSPECTIVE_EVALUATION_V1.md")]
    return {str(path.resolve()):sha(path) for path in paths}


def audit(project,output):
    from verify_historical_prediction import verify
    output = Path(output)
    if output.exists():raise FileExistsError("historical audit output must be fresh")
    with guarded() as counts:
        data,ids,manifests,original,cohort = load_historical(project)
        bindings = input_bindings(project)
        readback = verify(Path(project))
        previous = read(Path(project)/"research_artifacts"/ARCHIVE/"historical_readback.json")
        compare_metrics(readback["metrics"],previous["metrics"])
        archived = replay_archived_loo(Path(project)/"research_artifacts"/ARCHIVE,data,ids)
    result = dict(schema_version="hug_historical_prediction_audit_v1",audited_utc=datetime.now(timezone.utc).isoformat(),
        source_bindings={str(PROJECT/name):sha(PROJECT/name) for name in SOURCE_NAMES},input_bindings=bindings,
        source_cohort=cohort,datasets={name:manifest["counts"] for name,manifest in zip(DATASETS,manifests,strict=True)},
        historical_saved_weights=readback,archived_loo=archived,guard=counts,
        chronology="Later independent audit after the new-data model comparison; does not retroactively meet the brief's requested first-run order",
        new_historical_fits=0,new_controller_data_pooled=False,physical_inference_or_collection=0,
        remaining_brief_items=["Historical leave-one-failure-mode-out requires a separately declared risk-set design; pre-cutoff collisions cannot be treated as future failures."],
        scope="Immutable old weights and archived LOO prediction metrics only; historical inspected outcomes are not a new prospective test")
    for path,expected in bindings.items():
        if sha(path)!=expected:raise ValueError("historical input changed during replay")
    write(output,result)
    return result


def refit_declaration(project):
    import scipy
    from grasp_failure_prediction.prediction.retrospective_holdout import object_folds
    data,ids,manifests,_,cohort = load_historical(project)
    folds = object_folds(data["geometry_group_ids"],data["observation_group_ids"],data["group_ids"])
    return dict(schema_version="hug_historical_refit_protocol_v1",project=str(Path(project).resolve()),
        source_bindings={str(PROJECT/name):sha(PROJECT/name) for name in SOURCE_NAMES},input_bindings=input_bindings(project),
        source_cohort=cohort,runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        specification=dict(models=list(MODEL_NAMES),regularization_grid=[.001,.01,.1,1.],warning_threshold=.5,
            selection="validation_log_loss among converged candidates; train-only normalization",bootstrap_seed=20261015,bootstrap_repeats=1000),
        folds=[dict(held_geometry=fold["held_geometry"],partitions={role:[ids[i] for i in index]
                    for role,index in fold["indices"].items()}) for fold in folds],
        chronology="Optional later historical refit after the new-data comparison; not chronologically first or prospective",
        requires_commit_before_fit=True,optimization_calls_during_preparation=0,
        scope="Only original1040+2080 same-controller inspected data; no corrected/new-controller pooling or label/split changes")


def prepare_refit(project,output):
    if Path(output).exists():raise FileExistsError("historical refit declaration must be fresh")
    with guarded():declaration = refit_declaration(project)
    write(output,declaration)
    return declaration


def require_committed(protocol,declaration):
    paths = [Path(protocol).resolve(),*(Path(path) for path in declaration["source_bindings"])]
    relative = [str(path.relative_to(PROJECT)) for path in paths]
    subprocess.run(["git","ls-files","--error-unmatch","--",*relative],cwd=PROJECT,check=True,capture_output=True)
    subprocess.run(["git","diff","--quiet","HEAD","--",*relative],cwd=PROJECT,check=True,capture_output=True)
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip()


def run_refit(protocol,output):
    from grasp_failure_prediction.prediction.retrospective_holdout import run
    protocol,output = Path(protocol),Path(output)
    if output.exists():raise FileExistsError("historical refit output must be fresh")
    declaration = read(protocol)
    protocol_hash = sha(protocol)
    with guarded():
        if declaration!=refit_declaration(declaration["project"]):raise ValueError("historical refit declaration changed")
        committed = require_committed(protocol,declaration)
    with guarded(allow_fitting=True) as counts:
        report = run([Path(declaration["project"])/"research_artifacts"/name for name in DATASETS],output,protocol)
    if sha(protocol)!=protocol_hash:raise ValueError("historical refit protocol changed during fitting")
    for path,expected in {**declaration["source_bindings"],**declaration["input_bindings"]}.items():
        if sha(path)!=expected:raise ValueError("historical refit source/input changed during fitting")
    write(output/"refit_freeze_receipt.json",dict(protocol_sha256=protocol_hash,freeze_commit=committed,
                guard=counts,chronology=declaration["chronology"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=PROJECT)
    parser.add_argument("--runtime-project",type=Path,default=Path("/Users/nilakarthikesan/grasp-failure-prediction"))
    sub = parser.add_subparsers(dest="command",required=True)
    for command in ("audit","prepare-refit"):
        sub.add_parser(command).add_argument("--output",type=Path,required=True)
    refit = sub.add_parser("run-refit")
    refit.add_argument("--protocol",type=Path,required=True);refit.add_argument("--output",type=Path,required=True)
    args = parser.parse_args();bootstrap(args.runtime_project)
    if args.command=="audit":result = audit(args.project,args.output)
    elif args.command=="prepare-refit":result = prepare_refit(args.project,args.output)
    else:result = run_refit(args.protocol,args.output)
    print(json.dumps(dict(schema_version=result["schema_version"],output=str(args.output))))


if __name__=="__main__":main()
