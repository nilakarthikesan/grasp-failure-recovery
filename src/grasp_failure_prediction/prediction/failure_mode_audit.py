"""Separate failure-family holdouts and frozen category cluster uncertainty.

This audit does not revise historical or category-holdout results. Fitting is
explicit, summary-logistic only, and requires an exact committed declaration.
Cluster uncertainty reuses frozen probabilities, never fits or predicts anew.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy

from grasp_failure_prediction.evaluation.prediction_recording import PHYSICAL_FAILURES
from . import recorded_training as training
from .model import fit_logistic, outcome_metrics, save_model, load_model, summarize_prefix
from .retrospective_holdout import group_uncertainty


SCHEMA="hug_failure_family_audit_v1"
BOOTSTRAP_SCHEMA="hug_frozen_category_cluster_uncertainty_v1"
FAILURE_MODES=("approach_collision","no_contact","failed_acquisition","slip_during_hold","drop","timeout")
MODEL_KINDS=training.MODEL_NAMES
digest=training.canonical_sha256
sha=training.file_sha256


@dataclass(frozen=True)
class AuditSpec:
    model_kind: str="summary_logistic"
    regularization_grid: tuple[float,...]=(.001,.01,.1,1.)
    max_iterations: int=400
    validation_fraction: float=.2
    family_hash_seed: int=20261015
    warning_threshold: float=.5
    bootstrap_seed: int=20261015
    bootstrap_repeats: int=1000

    def __post_init__(self):
        # A named fixed audit, not a tunable model comparison or seed sweep.
        expected=dict(model_kind="summary_logistic",regularization_grid=(.001,.01,.1,1.),max_iterations=400,
            validation_fraction=.2,family_hash_seed=20261015,warning_threshold=.5,bootstrap_seed=20261015,bootstrap_repeats=1000)
        if (asdict(self)!=expected or any(type(getattr(self,name)) is not int for name in
                ("max_iterations","family_hash_seed","bootstrap_seed","bootstrap_repeats"))):
            raise ValueError("v1 audit has a fixed summary-only model/split/bootstrap budget")
        if frozenset(FAILURE_MODES)!=PHYSICAL_FAILURES:
            raise ValueError("v1 audit taxonomy differs from the bound physical scorer schema")


def _write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False);handle.write("\n")
    return sha(path)


def _source_files():
    package=Path(__file__).resolve().parents[1]
    names=("prediction/failure_mode_audit.py","prediction/model.py","prediction/recorded_training.py",
        "prediction/recorded_dataset.py","prediction/retrospective_holdout.py","evaluation/prediction_recording.py",
        "evaluation/scenario_sampler.py","evaluation/scoring.py")
    return {str(package/name):sha(package/name) for name in names}


def _verify(files):
    for filename,expected in files.items():
        if sha(Path(filename))!=expected:raise ValueError("bound audit source/input changed: "+filename)


def _jsonable(value):return json.loads(json.dumps(value,allow_nan=False))


def _attempt_inventory(data):
    training._check_loaded_inputs(data)
    scenarios={row["attempt_id"]:row for row in data.schedule["scenarios"]}
    # Legacy schedules can keep IDs only in a separate binding; this NEW audit
    # requires their explicit attempt IDs, as emitted by the frozen collector.
    if len(scenarios)!=len(data.schedule["scenarios"]):raise ValueError("explicit unique scheduled attempt IDs required")
    candidates=[Path(name) for name,hash_value in data.source_bindings["input_files"].items()
                if hash_value==data.source_bindings["binding_file_sha256"]]
    if len(candidates)!=1:raise ValueError("one exact attempt binding file required")
    binding=json.loads(candidates[0].read_text())
    if (digest(binding)!=data.source_bindings["attempt_binding_sha256"] or
            binding["attempt_to_scenario"]!={name:row["scenario_id"] for name,row in scenarios.items()}):
        raise ValueError("explicit scenario attempt IDs differ from the strict adapter binding")
    included=data.dataset.manifest["episodes"];excluded=data.dataset.manifest["excluded"]
    records=[]
    for index,row in enumerate(included):
        scenario=scenarios[row["attempt_id"]]
        failure=row["failure_type"]
        if failure is not None and failure not in FAILURE_MODES:raise ValueError("unknown physical failure cause")
        if (row["group_id"]!=scenario["proposal_group_id"] or row["provenance_hashes"]["scenario"]!=digest(scenario)):
            raise ValueError("eligible row is not bound to its scenario/family")
        records.append(dict(attempt_id=row["attempt_id"],family=scenario["proposal_group_id"],category=scenario["category_id"],
            eligible=True,index=index,success=int(data.dataset.y[index]),failure_type=failure,exclusion=None))
    for row in excluded:
        scenario=scenarios[row["attempt_id"]]
        failure=row.get("failure_type") if row["reason"]=="failure_at_or_before_cutoff" else None
        if row["reason"]=="failure_at_or_before_cutoff" and failure not in FAILURE_MODES:
            raise ValueError("excluded physical attempt lacks a valid cause")
        records.append(dict(attempt_id=row["attempt_id"],family=scenario["proposal_group_id"],category=scenario["category_id"],
            eligible=False,index=None,success=0 if failure is not None else None,failure_type=failure,exclusion=row["reason"]))
    if (len(records)!=len(scenarios) or {r["attempt_id"] for r in records}!=set(scenarios)):
        raise ValueError("audit must account for every planned attempt")
    return records


def failure_family_folds(data,*,specification=None):
    """A held cause removes its entire proposal families from fitting AND tuning."""
    spec=specification or AuditSpec();records=_attempt_inventory(data)
    families=sorted({r["family"] for r in records});folds=[]
    for held in FAILURE_MODES:
        events=[r for r in records if r["failure_type"]==held]
        held_families={r["family"] for r in events}
        remaining=set(families)-held_families
        ordered=sorted(remaining,key=lambda family:(digest(dict(audit=SCHEMA,seed=spec.family_hash_seed,held_failure=held,family=family)),family))
        n_validation=math.ceil(len(ordered)*spec.validation_fraction) if ordered else 0
        validation=set(ordered[:n_validation]);train=remaining-validation
        roles={family:("test" if family in held_families else "validation" if family in validation else "train") for family in families}
        partitions={}
        for role in ("train","validation","test"):
            chosen=[r for r in records if roles[r["family"]]==role]
            eligible=[r for r in chosen if r["eligible"]]
            partitions[role]=dict(families=sorted(f for f in families if roles[f]==role),
                attempt_ids=sorted(r["attempt_id"] for r in chosen),eligible_episode_ids=[r["attempt_id"] for r in eligible],
                row_indices=[r["index"] for r in eligible],planned=len(chosen),included=len(eligible),
                excluded=len(chosen)-len(eligible),success=sum(r["success"] for r in eligible),
                failure=sum(1-r["success"] for r in eligible),
                excluded_by_reason={reason:sum(r["exclusion"]==reason for r in chosen) for reason in sorted({r["exclusion"] for r in chosen if r["exclusion"]})})
        sets=[set(partitions[role]["families"]) for role in partitions]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError("proposal family leakage")
        if any(r["failure_type"]==held and roles[r["family"]]!="test" for r in records):raise ValueError("held failure leaked into fit/tune")
        cause_eligible=sum(r["eligible"] for r in events)
        reason="missing_failure_mode" if not events else "held_mode_has_no_eligible_prefix" if not cause_eligible else None
        if reason is None:
            reason=next(("empty_"+role for role,p in partitions.items() if not p["included"]),None)
        if reason is None:
            reason=next(("single_class_"+role for role,p in partitions.items() if not p["success"] or not p["failure"]),None)
        folds.append(dict(fold_id="held_failure_"+held,held_failure=held,status="SKIPPED" if reason else "READY",reason=reason,
            observed_cause_attempts=len(events),eligible_cause_attempts=cause_eligible,held_families=sorted(held_families),
            family_roles=roles,partitions=partitions,
            scope="binary task-outcome audit on held cause families; success controls and other co-occurring causes stay in test"))
    return dict(schema_version=SCHEMA,fixed_taxonomy=list(FAILURE_MODES),folds=folds,
        dataset_denominators=data.dataset.manifest["counts"],exclusion_policy="causally excluded/null attempts never become predictor rows",
        validation_rule="lowest ceil20percent remaining family hashes; no class balancing or replacement",outcome_dependent_family_holdout=True)


def _prediction_inventory(data,report_path):
    """Read frozen category probabilities only; never load or call a predictor."""
    training._check_loaded_inputs(data)
    report_path=Path(report_path).resolve();report=json.loads(report_path.read_text())
    if (report.get("schema_version")!=training.SCHEMA or report.get("status") not in {"completed","completed_with_skips","completed_with_model_errors"}
            or report.get("source_bindings")!=data.source_bindings or report.get("input_contract")!=data.dataset.manifest["input_contract"]
            or report["specification"]["warning_threshold"]!=.5):
        raise ValueError("frozen category report differs from this exact recorded cohort")
    expected={f["fold_id"]:f for f in data.folds}
    if len(report["folds"])!=len(expected) or {f["fold_id"] for f in report["folds"]}!=set(expected):raise ValueError("category report dropped/repeated a fold")
    files={str(report_path):sha(report_path)};rows=[]
    for fold in report["folds"]:
        actual=expected[fold["fold_id"]]
        if any(fold[key]!=actual[key] for key in ("test_category","validation_category")):raise ValueError("held category identity changed")
        for role,indices in actual["indices"].items():
            partition=fold["partitions"][role]
            if (partition["episode_ids"]!=data.dataset.episode_ids[indices].tolist() or partition["n"]!=len(indices)
                    or partition["row_indices_sha256"]!=digest(indices.tolist())):raise ValueError("category partition rows differ from frozen folds")
        inventory={(name,seed) for name in MODEL_KINDS for seed in (0,1,2)}
        if len(fold["models"])!=15 or {(r["name"],r["seed"]) for r in fold["models"]}!=inventory:raise ValueError("must retain every original model/seed entry")
        test=actual["indices"]["test"];ids=data.dataset.episode_ids[test].tolist();truth=data.dataset.y[test]
        for model in fold["models"]:
            row=dict(fold_id=fold["fold_id"],category=fold["test_category"],model_kind=model["name"],seed=model["seed"],
                status=model["status"],reason=model.get("reason"),episode_ids=ids,families=data.dataset.group_ids[test].tolist())
            if model["status"]=="completed":
                root=(report_path.parent/model["model_directory"]).resolve()
                if not root.is_relative_to(report_path.parent):raise ValueError("prediction bundle escapes report root")
                path=root/"predictions.json"
                if sha(path)!=model["predictions_sha256"]:raise ValueError("frozen category prediction hash mismatch")
                prediction=json.loads(path.read_text())
                if (prediction["episode_ids"]!=ids or prediction["y_success"]!=truth.tolist() or prediction["warning_threshold"]!=.5):
                    raise ValueError("frozen prediction labels/order/threshold differ")
                probability=np.asarray(prediction["p_success"],float);outcome_metrics(truth,probability)
                files[str(path)]=sha(path)
                row.update(predictions_path=str(path),predictions_sha256=sha(path))
            elif model["status"] not in {"skipped","error"}:raise ValueError("unknown original model status")
            rows.append(row)
    return dict(report_path=str(report_path),entries=rows,files=files)


def _runtime():return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)


def prepare_audit(data,prediction_report,protocol_path,*,specification=None):
    spec=specification or AuditSpec()
    folds=failure_family_folds(data,specification=spec);predictions=_prediction_inventory(data,prediction_report)
    value=dict(schema_version=SCHEMA,specification=_jsonable(asdict(spec)),source_files=_source_files(),
        input_files={**data.source_bindings["input_files"],**predictions["files"]},
        source_bindings=data.source_bindings,input_contract=data.dataset.manifest["input_contract"],cohort=data.dataset.manifest["cohort"],
        failure_fold_plan=folds,category_prediction_inventory=predictions,runtime=_runtime(),requires_commit_before_fit=True,
        analysis_scope="separate label-defined failure-family summary-logistic audit; not prospective unseen-mode detection or a five-model comparison",
        bootstrap_scope="frozen category predictions; conditional family clusters within each held category; seeds are reported separately, never resampled/pooled")
    _write(protocol_path,value)
    return value


def _committed(protocol,files):
    path=Path(protocol).resolve()
    root=Path(subprocess.check_output(["git","-C",str(path.parent),"rev-parse","--show-toplevel"],text=True).strip()).resolve()
    relative=[str(p.relative_to(root)) for p in [path,*map(Path,files)]]
    subprocess.run(["git","-C",str(root),"ls-files","--error-unmatch","--",*relative],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    subprocess.run(["git","-C",str(root),"diff","--quiet","HEAD","--",*relative],check=True)
    return dict(commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),bound_protocol_and_sources_clean=True)


def validate_audit(data,protocol_path,*,require_commit=True):
    value=json.loads(Path(protocol_path).read_text());spec=AuditSpec(**{**value["specification"],"regularization_grid":tuple(value["specification"]["regularization_grid"])})
    training._check_loaded_inputs(data)
    if (value.get("schema_version")!=SCHEMA or value.get("requires_commit_before_fit") is not True
            or value["source_files"]!=_source_files() or value["source_bindings"]!=data.source_bindings
            or value["input_contract"]!=data.dataset.manifest["input_contract"] or value["cohort"]!=data.dataset.manifest["cohort"]
            or value["runtime"]!=_runtime() or value["failure_fold_plan"]!=failure_family_folds(data,specification=spec)
            or value["category_prediction_inventory"]!=_prediction_inventory(data,value["category_prediction_inventory"]["report_path"])):
        raise ValueError("frozen failure audit source/data/split/prediction contract changed")
    _verify({**value["source_files"],**value["input_files"]})
    expected_inputs={**data.source_bindings["input_files"],**value["category_prediction_inventory"]["files"]}
    if value["input_files"]!=expected_inputs:raise ValueError("audit input file closure differs from its exact data/predictions")
    receipt=_committed(protocol_path,value["source_files"]) if require_commit else None
    return value,spec,receipt


def frozen_category_uncertainty(data,protocol,*,specification=None):
    """Reuse the existing historical family-cluster algorithm on NEW strict data."""
    spec=specification or AuditSpec()
    declaration_sha256=digest(protocol)
    if (protocol["source_files"]!=_source_files() or protocol["source_bindings"]!=data.source_bindings
            or protocol["input_contract"]!=data.dataset.manifest["input_contract"]
            or protocol["cohort"]!=data.dataset.manifest["cohort"]):
        raise ValueError("cluster uncertainty declaration differs from this source/cohort")
    _verify(protocol["input_files"]);training._check_loaded_inputs(data)
    inventory=_prediction_inventory(data,protocol["category_prediction_inventory"]["report_path"])
    if inventory!=protocol["category_prediction_inventory"]:raise ValueError("frozen prediction inventory changed")
    by_id={name:i for i,name in enumerate(data.dataset.episode_ids.tolist())}
    rows=[]
    for entry in inventory["entries"]:
        row=dict(entry);rows.append(row)
        if entry["status"]!="completed":row.update(status="SKIPPED_ORIGINAL_MODEL",uncertainty=None);continue
        indices=np.asarray([by_id[name] for name in entry["episode_ids"]],int)
        probability=np.asarray(json.loads(Path(entry["predictions_path"]).read_text())["p_success"],float)
        y=data.dataset.y[indices];families=data.dataset.group_ids[indices]
        if not len(y):row.update(status="SKIPPED_EMPTY_CATEGORY",uncertainty=None);continue
        metrics=outcome_metrics(y,probability)
        if len(np.unique(y))<2:metrics.update(failure_roc_auc=None,failure_average_precision=None)
        row["point_metrics"]=metrics
        if len(np.unique(families))<2:
            row.update(status="SKIPPED_TOO_FEW_FAMILIES",uncertainty=None);continue
        uncertainty=group_uncertainty(y,probability,families,seed=spec.bootstrap_seed,repeats=spec.bootstrap_repeats)
        uncertainty["unit"]="whole proposal families, conditional on this held category"
        for item in uncertainty["percentile_95"].values():item["undefined_repeats"]=spec.bootstrap_repeats-item["valid_repeats"]
        row.update(status="COMPLETED",uncertainty=uncertainty)
    # The calculation can outlast an external file edit. Never return a result
    # unless the original in-memory declaration and every bound byte still hold.
    if digest(protocol)!=declaration_sha256:raise ValueError("uncertainty declaration changed during calculation")
    _verify({**protocol["source_files"],**protocol["input_files"]})
    training._check_loaded_inputs(data)
    if _prediction_inventory(data,inventory["report_path"])!=inventory:
        raise ValueError("frozen prediction inventory changed during calculation")
    return dict(schema_version=BOOTSTRAP_SCHEMA,scope=protocol["bootstrap_scope"],rows=rows,
        source_dataset_denominators=data.dataset.manifest["counts"],fitting_calls=0,prediction_calls=0,
        seed_repetitions_are_independent_datasets=False,estimator_source_sha256=sha(Path(__file__).with_name("retrospective_holdout.py")))


def run_audit(data,protocol_path,output):
    value,spec,receipt=validate_audit(data,protocol_path)
    output=Path(output)
    if output.exists():raise FileExistsError("new audit output directory required; no retries/overwrite")
    output.mkdir(parents=True)
    report=dict(schema_version=SCHEMA,protocol_sha256=sha(Path(protocol_path)),frozen_receipt=receipt,
        model_kind=spec.model_kind,scope=value["analysis_scope"],folds=[],dataset_denominators=data.dataset.manifest["counts"],status="COMPLETED",
        source_bindings=value["source_bindings"],source_files=value["source_files"],input_contract=value["input_contract"],cohort=value["cohort"],
        planned_candidate_fit_budget=4*sum(f["status"]=="READY" for f in value["failure_fold_plan"]["folds"]),actual_candidate_fits=0)
    contract=data.dataset.manifest["input_contract"]
    features=summarize_prefix(data.dataset.X,history_steps=contract["history_steps"],joint_count=len(contract["joint_names"]))
    hard_error=None
    for fold in value["failure_fold_plan"]["folds"]:
        row=dict(fold);report["folds"].append(row)
        if fold["status"]=="SKIPPED":continue
        if hard_error:row.update(status="NOT_RUN",reason="earlier_audit_error");continue
        try:
            train,validation,test=(np.asarray(fold["partitions"][role]["row_indices"],int) for role in ("train","validation","test"))
            candidates=[];models=[]
            for regularization in spec.regularization_grid:
                validate_audit(data,protocol_path)
                report["actual_candidate_fits"]+=1
                model=fit_logistic(features[train],data.dataset.y[train],regularization=regularization,max_iterations=spec.max_iterations)
                p=model.predict_success(features[validation]);m=outcome_metrics(data.dataset.y[validation],p)
                candidates.append(dict(regularization=regularization,converged=bool(model.converged),iterations=model.iterations,validation_log_loss=m["log_loss"]))
                models.append(model)
            eligible=[i for i,m in enumerate(models) if m.converged]
            if not eligible:raise RuntimeError("no summary-logistic candidate converged")
            chosen=min(eligible,key=lambda i:(candidates[i]["validation_log_loss"],-spec.regularization_grid[i]))
            model=models[chosen]
            validate_audit(data,protocol_path)
            probability=model.predict_success(features[test])
            metrics=outcome_metrics(data.dataset.y[test],probability)
            folder=output/fold["fold_id"];folder.mkdir()
            save_model(model,folder/"model.npz")
            if not np.allclose(load_model(folder/"model.npz").predict_success(features[test]),probability,rtol=0,atol=1e-12):
                raise RuntimeError("saved audit model readback differs")
            predictions=dict(episode_ids=data.dataset.episode_ids[test].tolist(),y_success=data.dataset.y[test].tolist(),p_success=probability.tolist(),warning_threshold=.5)
            pred_sha=_write(folder/"predictions.json",predictions)
            row.update(status="COMPLETED",selection=dict(candidates=candidates,selected_candidate=chosen,tie_break="stronger L2",selection_metric="validation_log_loss"),
                metrics=metrics,model_sha256=sha(folder/"model.npz"),predictions_sha256=pred_sha)
            validate_audit(data,protocol_path)
        except Exception as error:
            hard_error=dict(type=type(error).__name__,message=str(error));row.update(status="ERROR",error=hard_error)
            report.update(status="HARD_STOP",error=hard_error)
    try:
        report["category_cluster_uncertainty"]=frozen_category_uncertainty(data,value,specification=spec)
        validate_audit(data,protocol_path)
    except Exception as error:
        report.update(status="HARD_STOP",error=dict(type=type(error).__name__,message=str(error)))
    _write(output/"report.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True);parser.add_argument("--schedule",type=Path,required=True)
    parser.add_argument("--category-folds",type=Path,required=True);parser.add_argument("--attempt-binding",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare");p.add_argument("--category-report",type=Path,required=True)
    p=sub.add_parser("run");p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("uncertainty");p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    data=training.load_training_data(args.dataset,args.schedule,args.category_folds,args.attempt_binding)
    if args.command=="prepare":result=prepare_audit(data,args.category_report,args.protocol)
    elif args.command=="run":result=run_audit(data,args.protocol,args.output)
    else:
        if args.output.exists():raise FileExistsError("new uncertainty output file required; no retries/overwrite")
        protocol,spec,_=validate_audit(data,args.protocol)
        declaration_sha256=sha(args.protocol)
        result=frozen_category_uncertainty(data,protocol,specification=spec)
        validate_audit(data,args.protocol)
        if sha(args.protocol)!=declaration_sha256:raise ValueError("uncertainty protocol changed during calculation")
        _write(args.output,result)
    print(json.dumps(dict(schema_version=result["schema_version"],status=result.get("status","PREPARED_OR_MEASURED"))))


if __name__=="__main__":main()
