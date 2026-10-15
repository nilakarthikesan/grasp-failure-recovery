#!/usr/bin/env python3
"""Prepared recorded-cohort cutoff diagnostic; never fit or simulate.

The fixed 2.96/2.76/2.56/2.36 s grid retains 20 x 24 x 3 causal close samples.
Earlier cutoffs deliberately change the input protocol; their scores are a
shifted-time diagnostic, not deployment-equivalent validation of the fitted
2.96 s contract. Only original held-test episode IDs can reach each predictor.
Known first-failure timestamps are recorded proxies, never slip-onset truth.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import mujoco

import run_hug_build_collection as collector
import evaluate_hug_reserve_models as frozen

SCHEMA = "hug_recorded_cutoff_sensitivity_protocol_v1"
GRID = (2.96, 2.76, 2.56, 2.36)
PRIMARY = "summary_logistic"
OWN_SOURCE = "scripts/audit_hug_cutoff_sensitivity.py"
# Only availability/phase causes are exclusions. Interval, endpoint, finite,
# identity, source hash and all unexpected exporter errors are hard stops.
UNAVAILABLE_ERRORS = frozenset({"insufficient samples before prediction cutoff",
                              "prediction history must be closing, before any lift/hold/score"})


def _json(path):
    return json.loads(Path(path).read_text())


def cutoff_key(value):
    return f"t{round(value * 100):03d}"


def _source_files(project):
    paths = (*collector.SOURCE_FILES, *frozen.EXTRA_SOURCES, OWN_SOURCE)
    return {str(Path(project) / name): collector.sha(Path(project) / name) for name in paths}


def _manifest_path(report):
    paths = [Path(path) for path in report["source_bindings"]["input_files"] if Path(path).name == "manifest.json"
             and _json(path).get("schema_version") == "hug_recorded_prefix_dataset_v1"]
    if len(paths) != 1:
        raise ValueError("the original fitted dataset manifest must be uniquely bound")
    return paths[0]


def _folds(report, models):
    """Bind original partitions and every completed bundle's saved test rows."""
    result = {}
    for fold in report["folds"]:
        ids = {role: fold["partitions"][role]["episode_ids"] for role in ("train", "validation", "test")}
        if (fold["fold_id"] in result or any(len(value) != len(set(value)) for value in ids.values())
                or any(set(ids[a]) & set(ids[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))):
            raise ValueError("original model partitions repeat or overlap episode IDs")
        result[fold["fold_id"]] = ids
    for bundle in models["bundles"]:
        ids = result[bundle["fold_id"]]
        directory = Path(bundle["directory"])
        metadata, predictions = _json(directory / "model.json"), _json(directory / "predictions.json")
        if (metadata.get("training_episode_ids_sha256") != collector.digest(ids["train"])
                or metadata.get("validation_episode_ids_sha256") != collector.digest(ids["validation"])
                or predictions["episode_ids"] != ids["test"] or predictions["warning_threshold"] != .5):
            raise ValueError("model identity or saved test predictions differ from original partitions")
    return result


def prepare(project, main_preparation, training_protocol, model_report, study_summary, output):
    """Hash/read only; no alternate prefix extraction, exports or predictions."""
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("cutoff evidence requires a fresh directory")
    main_path, study_path = Path(main_preparation).resolve(), Path(study_summary).resolve()
    main, study = _json(main_path), _json(study_path)
    collector.verify({**main["source_bindings"], **main["input_bindings"], **main["asset_bindings"]})
    collector.bootstrap(main)
    report, models = frozen.complete_model_bundles(model_report)
    closure = frozen.verify_fitted_source_closure(project, report, training_protocol)
    cohort = frozen.verify_training_cohort(project, report)
    contract = models["input_contract"]
    if (contract["prediction_time_s"] != GRID[0] or contract["history_steps"] != 20
            or len(contract["joint_names"]) != 24 or contract["control_timestep_s"] != .04
            or contract["features"] != 1440):
        raise ValueError("cutoff audit requires the unchanged original 2.96/20/24/3 contract")
    if (study.get("condition") != "study" or study.get("status") != "COMPLETED_BOUNDED_QUOTA"
            or study["preparation_sha256"] != collector.sha(main_path)
            or len(study["attempts"]) != study["quota"]):
        raise ValueError("a complete original study ledger is required")
    collector.verify(study["raw_artifact_bindings"])
    roots = [{"attempt_id": row["attempt_id"], "recording_path": str(Path(row["recording_path"]).resolve())}
             for row in study["attempts"]]
    if (not roots or len({row["attempt_id"] for row in roots}) != len(roots)
            or len({row["recording_path"] for row in roots}) != len(roots)):
        raise ValueError("study attempt identity/recording paths repeat")
    for row in roots:
        if str(Path(row["recording_path"]) / "attempt.json") not in study["raw_artifact_bindings"]:
            raise ValueError("every raw recording must be hash-bound by the original study")
    manifest = _manifest_path(report)
    original = _json(manifest)
    if {row["attempt_id"] for row in original["episodes"] + original["excluded"]} != {row["attempt_id"] for row in roots}:
        raise ValueError("original fitted dataset and study attempt denominators differ")
    folds = _folds(report, models)
    eligible = {row["attempt_id"] for row in original["episodes"]}
    if any(not set(ids[role]) <= eligible for ids in folds.values() for role in ids):
        raise ValueError("original fold membership escapes fitted dataset episode IDs")
    declaration = dict(schema_version=SCHEMA, status="PREPARED_NOT_RUN", project=str(project),
        main_preparation_path=str(main_path),study_summary_path=str(study_path),
        dependency_dirs=main["dependency_dirs"], runtime=collector.runtime(),
        cutoffs_s=list(GRID), warning_threshold=.5, primary_model=PRIMARY,
        models=models, original_folds=folds, fitted_source_closure=closure, expected_cohort=cohort,
        original_dataset=str(manifest.parent), attempts=roots,
        source_bindings=_source_files(project),
        input_bindings={str(main_path):collector.sha(main_path), str(study_path):collector.sha(study_path),
            str(Path(training_protocol).resolve()):collector.sha(training_protocol),
            **report["source_bindings"]["input_files"], **study["raw_artifact_bindings"]},
        history_steps=20, joint_count=24, field_count=3, control_timestep_s=.04,
        fit_allowed=False, physics_allowed=False, threshold_tuning_allowed=False, cutoff_selection_allowed=False,
        baseline_readback_atol=dict(small_gru=1e-7, other=1e-12),
        scope="new shifted-time input protocols; original test IDs only; no deployment-equivalent earlier-cutoff claim",
        proxy_scope="first_failure_time_s where recorded; no inferred/imputed slip onset or new labels")
    collector.write(output / "protocol.draft.json", declaration)
    return declaration


def _verify(protocol):
    if (protocol.get("schema_version") != SCHEMA or protocol.get("cutoffs_s") != list(GRID)
            or protocol.get("primary_model") != PRIMARY or protocol.get("warning_threshold") != .5
            or any(protocol.get(key) is not False for key in ("fit_allowed", "physics_allowed", "threshold_tuning_allowed", "cutoff_selection_allowed"))
            or (protocol["history_steps"], protocol["joint_count"], protocol["field_count"], protocol["control_timestep_s"]) != (20,24,3,.04)
            or protocol["baseline_readback_atol"] != dict(small_gru=1e-7,other=1e-12)
            or protocol["runtime"] != collector.runtime()):
        raise ValueError("cutoff protocol grid, fixed policy or runtime changed")
    if _source_files(protocol["project"]) != protocol["source_bindings"]:
        raise ValueError("cutoff bound source inventory changed")
    collector.verify({**protocol["source_bindings"], **protocol["input_bindings"], **protocol["models"]["bindings"]})
    collector.bootstrap(protocol)
    report, models = frozen.complete_model_bundles(protocol["models"]["report_path"])
    main_path,study_path=Path(protocol["main_preparation_path"]),Path(protocol["study_summary_path"])
    main,study=_json(main_path),_json(study_path)
    roots=[dict(attempt_id=row["attempt_id"],recording_path=str(Path(row["recording_path"]).resolve())) for row in study["attempts"]]
    bindings={str(main_path):collector.sha(main_path),str(study_path):collector.sha(study_path),
        protocol["fitted_source_closure"]["protocol_path"]:protocol["fitted_source_closure"]["protocol_sha256"],
        **report["source_bindings"]["input_files"],**study["raw_artifact_bindings"]}
    if (protocol["attempts"]!=roots or protocol["input_bindings"]!=bindings
            or protocol["dependency_dirs"]!=main["dependency_dirs"]
            or protocol["original_dataset"]!=str(_manifest_path(report).parent)
            or study.get("condition")!="study" or study.get("status")!="COMPLETED_BOUNDED_QUOTA"
            or len(roots)!=study["quota"] or len({row["attempt_id"] for row in roots})!=len(roots)
            or study["preparation_sha256"]!=collector.sha(main_path)):
        raise ValueError("cutoff original study/data/attempt inventory changed")
    collector.verify({**main["source_bindings"],**main["input_bindings"],**main["asset_bindings"]})
    if (models != protocol["models"] or _folds(report, models) != protocol["original_folds"]
            or frozen.verify_fitted_source_closure(protocol["project"],report,protocol["fitted_source_closure"]["protocol_path"]) != protocol["fitted_source_closure"]
            or frozen.verify_training_cohort(protocol["project"], report) != protocol["expected_cohort"]):
        raise ValueError("cutoff model/source/cohort/partition identity changed")


def freeze(output):
    output = Path(output).resolve()
    declaration = _json(output / "protocol.draft.json"); _verify(declaration)
    collector.write(output / "protocol.json", declaration)
    return collector.sha(output / "protocol.json")


def read_protocol(output):
    path = Path(output).resolve() / "protocol.json"
    protocol = _json(path); _verify(protocol)
    frozen._require_commit([path, *protocol["source_bindings"]])
    return protocol


def _spec(protocol, cutoff):
    from grasp_failure_prediction.prediction.recorded_dataset import PrefixSpec
    base = protocol["models"]["input_contract"]
    return PrefixSpec(protocol_id=base["protocol_id"] if cutoff == GRID[0] else f"hug_close_{cutoff_key(cutoff)}_shifted_diagnostic_v1",
        prediction_time_s=cutoff,history_steps=20,control_timestep_s=.04)


def _export_cutoff(protocol, cutoff, output):
    """The wrapper retains all rows; strict export sees only available rows."""
    from grasp_failure_prediction.evaluation.prediction_recording import load_attempt
    from grasp_failure_prediction.prediction.recorded_dataset import _prefix,export_recorded_dataset,load_recorded_dataset
    spec, rows, included = _spec(protocol, cutoff), [], []
    for source in protocol["attempts"]:
        attempt, payload, arrays = load_attempt(source["recording_path"])
        if attempt.attempt_id != source["attempt_id"]:
            raise ValueError("raw attempt identity differs from frozen study")
        row = dict(**source,status="EXCLUDED",reason=None,first_failure_time_s=None)
        if attempt.mapping.status.value != "ok": row["reason"]="mapping_refusal"
        elif attempt.execution_status != "completed" or attempt.outcome is None: row["reason"]="unexecuted_or_runtime_error"
        else:
            outcome=attempt.outcome
            cohort=dict(hand_id=attempt.hand_id,execution_protocol_id=attempt.execution_protocol_id,
                        label_rule_id=outcome.label_rule_id,label_rule_sha256=attempt.provenance_hashes["label_rule"])
            if cohort != protocol["expected_cohort"]:
                raise ValueError("raw physical cohort differs from frozen fitting cohort")
            row.update(first_failure_time_s=outcome.first_failure_time_s,success=outcome.success)
            if arrays["joint_names"].astype(str).tolist() != protocol["models"]["input_contract"]["joint_names"]:
                raise ValueError("raw joint ordering differs from original model")
            if not np.allclose(np.diff(arrays["time_s"]),.04,rtol=0,atol=1e-9):
                raise ValueError("raw recording sample alignment changed")
            if outcome.failure_type == "approach_collision" or (outcome.first_failure_time_s is not None and outcome.first_failure_time_s <= cutoff+1e-9):
                row["reason"]="failure_at_or_before_cutoff"
            else:
                try:
                    _prefix(arrays,spec)
                except ValueError as error:
                    if str(error) not in UNAVAILABLE_ERRORS: raise
                    row.update(status="UNAVAILABLE",reason=str(error))
                else:
                    row.update(status="INCLUDED",reason=None);included.append(source["recording_path"])
        rows.append(row)
    info=dict(cutoff_s=cutoff, input_protocol=spec.model_dump(mode="json"),attempted=len(rows),
        included=len(included),exclusion_counts=dict(Counter(row["reason"] for row in rows if row["status"]!="INCLUDED")),audit_rows=rows,
        status="EXPORTED" if included else "UNDEFINED_NO_ELIGIBLE_PREFIXES",dataset=None)
    if included:
        root=Path(output)/cutoff_key(cutoff)
        manifest=export_recorded_dataset(included,root,prefix_spec=spec)
        dataset=load_recorded_dataset(root)
        original=dict(protocol["models"]["input_contract"])
        expected={**original,**spec.model_dump(mode="json")}
        if manifest["input_contract"] != expected or manifest["cohort"] != protocol["expected_cohort"]:
            raise ValueError("strict export changed input allowlist or physical cohort")
        info["dataset"]=dict(directory=str(root),bindings=frozen._files(root))
        return info,dataset
    return info,None


def _lead_summary(rows, ids):
    selected=[row for row in rows if row["attempt_id"] in set(ids)]
    known=[row["first_failure_time_s"] for row in selected if row.get("success") is False and row["first_failure_time_s"] is not None]
    return dict(scope="recorded first-failure proxy only; not slip onset",selected=len(selected),known_proxy_count=len(known),
                missing_failure_proxy_count=sum(row.get("success") is False and row["first_failure_time_s"] is None for row in selected),
                first_failure_times_s=known)


def run(output):
    """Explicit one-pass export/prediction after the declaration is committed."""
    output=Path(output).resolve();protocol=read_protocol(output)
    if (output/"started.json").exists() or (output/"report.json").exists():
        raise FileExistsError("cutoff audit cannot retry or overwrite")
    collector.write(output/"started.json",dict(protocol_sha256=collector.sha(output/"protocol.json")))
    report=dict(schema_version="hug_recorded_cutoff_sensitivity_v1",status="STARTED",primary_model=PRIMARY,warning_threshold=.5,
        protocol_sha256=collector.sha(output/"protocol.json"),fitting_calls=0,physics_calls=0,
        scope=protocol["scope"],proxy_scope=protocol["proxy_scope"],
        cutoffs=[dict(cutoff_s=cutoff,status="NOT_RUN") for cutoff in GRID],
        models=[dict(**bundle,status="NOT_RUN",cutoffs=[dict(cutoff_s=cutoff,status="NOT_RUN",episode_ids=[],metrics=None,common_metrics=None) for cutoff in GRID])
                for bundle in protocol["models"]["bundles"]])
    try:
        from grasp_failure_prediction.prediction.recorded_dataset import load_recorded_dataset
        from grasp_failure_prediction.prediction.recorded_training import predict_saved_model
        original=load_recorded_dataset(protocol["original_dataset"])
        datasets={}
        for i,cutoff in enumerate(GRID):
            report["cutoffs"][i]["status"]="STARTED"
            info,data=_export_cutoff(protocol,cutoff,output/"prefixes")
            report["cutoffs"][i]=info;datasets[cutoff]=data
        baseline=datasets[GRID[0]]
        if baseline is None or any(not np.array_equal(getattr(baseline,key),getattr(original,key)) for key in (
                "X","y","episode_ids","group_ids","observation_group_ids","geometry_group_ids")):
            raise ValueError("baseline 2.96 strict export does not exactly reproduce original fitted arrays/IDs")
        report["baseline_exact_arrays"] = True
        for record in report["models"]:
            held=protocol["original_folds"][record["fold_id"]]["test"]
            available={cutoff: ([] if data is None else [value for value in held if value in set(data.episode_ids.tolist())]) for cutoff,data in datasets.items()}
            common=[value for value in held if all(value in available[cutoff] for cutoff in GRID)]
            record.update(status="STARTED",original_test_episode_ids=held,common_episode_ids=common,common_episode_ids_sha256=collector.digest(common))
            for cutoff in GRID:
                ids=available[cutoff]
                cell=record["cutoffs"][GRID.index(cutoff)]
                cell.update(status="UNDEFINED_NO_ORIGINAL_TEST_PREFIXES",episode_ids=ids)
                lead=_lead_summary(report["cutoffs"][GRID.index(cutoff)]["audit_rows"],ids)
                values=np.asarray(lead.pop("first_failure_times_s"),dtype=float)-cutoff
                lead.update(lead_s=values.tolist(),minimum_s=float(values.min()) if len(values) else None,
                    median_s=float(np.median(values)) if len(values) else None,maximum_s=float(values.max()) if len(values) else None)
                cell["known_first_failure_proxy_lead"]=lead
                if not ids:continue
                cell["status"]="STARTED"
                data=datasets[cutoff];lookup={value:i for i,value in enumerate(data.episode_ids.tolist())};index=np.asarray([lookup[value] for value in ids])
                probability=np.asarray(predict_saved_model(record["directory"],data.X[index]),dtype=float)
                metrics=frozen._metrics(data.y[index],probability)
                if cutoff==GRID[0]:
                    saved=_json(Path(record["directory"])/"predictions.json")
                    atol=protocol["baseline_readback_atol"]["small_gru" if record["kind"]=="small_gru" else "other"]
                    if saved["episode_ids"]!=ids or saved["y_success"]!=data.y[index].tolist() or not np.allclose(saved["p_success"],probability,rtol=0,atol=atol):
                        raise ValueError("baseline fixed-model prediction readback differs from original test results")
                common_index=np.asarray([ids.index(value) for value in common],dtype=int)
                pred=dict(episode_ids=ids,y_success=data.y[index].tolist(),p_success=probability.tolist(),warning_threshold=.5,
                    actual_input_contract=data.manifest["input_contract"],fitted_input_contract=protocol["models"]["input_contract"],
                    diagnostic_input_shift=cutoff!=GRID[0],X_sha256=collector.digest(dict(dtype=data.X[index].dtype.str,shape=list(data.X[index].shape),data_sha256=hashlib.sha256(data.X[index].tobytes()).hexdigest())))
                path=output/"predictions"/record["bundle_id"]/(cutoff_key(cutoff)+".json");collector.write(path,pred)
                cell.update(status="COMPLETED",metrics=metrics,common_metrics=frozen._metrics(data.y[index][common_index],probability[common_index]) if common else None,
                    predictions_sha256=collector.sha(path),diagnostic_input_shift=cutoff!=GRID[0])
            record["status"]="COMPLETED"
        report["status"]="COMPLETED_DIAGNOSTIC"
        _verify(protocol)
    except Exception as error:
        report.update(status="HARD_STOP",error=dict(type=type(error).__name__,message=str(error)))
        (output/"traceback.txt").write_text(traceback.format_exc())
        for cutoff in report["cutoffs"]:
            if cutoff["status"]=="STARTED":cutoff["status"]="ERROR"
        for record in report["models"]:
            if record["status"]=="STARTED":record["status"]="ERROR"
            for cell in record["cutoffs"]:
                if cell["status"]=="STARTED":cell["status"]="ERROR"
    report["derived_artifact_bindings"]={**frozen._files(output/"prefixes"),**frozen._files(output/"predictions")}
    collector.write(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare")
    for name in ("project","main-preparation","training-protocol","model-report","study-summary","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for command in ("freeze","run"):
        commands.add_parser(command).add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="prepare":result=prepare(args.project,args.main_preparation,args.training_protocol,args.model_report,args.study_summary,args.output)
    elif args.command=="freeze":result=dict(status="FROZEN_NOT_RUN",sha256=freeze(args.output))
    else:result=run(args.output)
    print(json.dumps(dict(status=result["status"])))


if __name__=="__main__":main()
