"""Native MANOSim pre-lift outcome prediction; no collection or automatic fits.

Scientific fitting requires a frozen native task qualification and independent
readback, never capsule fixtures/Shadow data. Synthetic tests are software only.
Raw wxyz ball coordinates are hemisphere-canonicalized by sign only; scene
addresses, pose-to-control conversion, contacts and outcomes are not features.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator

from grasp_failure_prediction.evaluation.manosim_recording import (
    NATIVE_DOMAIN, QPOS_NAMES, QVEL_NAMES, CTRL_NAMES, NativeRecordingContract,
    canonical_sha256, file_sha256, load_native_attempt, validate_native_observables,
)
from grasp_failure_prediction.evaluation.prediction_recording import MappingStatus
from .model import Normalizer, fit_logistic, outcome_metrics

SCHEMA = "manosim_native_training_v1"
MODEL_SCHEMA = "manosim_native_pre_lift_model_v1"
MODELS = ("training_prevalence", "summary_logistic", "summary_mlp32", "temporal_gru16")
IDENTITIES = ("category_id", "object_id", "geometry_group_id", "observation_group_id", "proposal_group_id")
FIELDS = ("hand_qpos_hemisphere", "hand_qvel", "effective_ctrl")
ROLES = ("train", "validation", "test")
_HASH = __import__("re").compile(r"sha256:[a-f0-9]{64}")


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    def reject(value): raise ValueError("nonfinite JSON")
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject)
    def finite(x):
        if isinstance(x, float) and not np.isfinite(x): raise ValueError("nonfinite JSON")
        if isinstance(x, dict):
            for y in x.values(): finite(y)
        if isinstance(x, list):
            for y in x: finite(y)
    finite(value)
    return value


def write(path, value):
    path = Path(path)
    with path.open("x") as f: json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write("\n")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Binding(StrictModel):
    path: StrictStr
    sha256: StrictStr

    @field_validator("sha256")
    @classmethod
    def digest(cls, value):
        if not _HASH.fullmatch(value): raise ValueError("tagged SHA-256 required")
        return value

    def verify(self):
        p = Path(self.path)
        if not p.is_absolute() or p.resolve() != p or any(x.is_symlink() for x in (p, *p.parents)) or not p.is_file() or file_sha256(p) != self.sha256:
            raise ValueError("bound input absent/changed/aliased: " + self.path)
        return p


def binding(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": file_sha256(p)}


class PrefixSpec(StrictModel):
    protocol_id: StrictStr = "native_close20_lead1_v1"
    history_steps: Literal[20] = 20
    lead_intervals: Literal[1] = 1


class TrainingSpec(StrictModel):
    seeds: tuple[StrictInt, ...] = (0, 1, 2)
    logistic_l2: tuple[float, ...] = (.001, .01, .1, 1.)
    neural_learning_rates: tuple[float, ...] = (.001, .003)
    epochs: StrictInt = Field(default=50, ge=1, le=50)
    logistic_iterations: StrictInt = Field(default=400, ge=1, le=400)
    weight_decay: float = Field(default=.0001, ge=0, allow_inf_nan=False)
    threshold: Literal[.5] = .5

    @model_validator(mode="after")
    def bounded(self):
        if self.seeds != (0, 1, 2) or self.logistic_l2 != (.001, .01, .1, 1.) or self.neural_learning_rates != (.001, .003) or self.weight_decay != .0001:
            raise ValueError("v1 fixed seeds/candidates/regularization required")
        return self


class ScheduledAttempt(StrictModel):
    attempt_id: StrictStr
    category_id: StrictStr
    object_id: StrictStr
    geometry_group_id: StrictStr
    observation_group_id: StrictStr
    proposal_group_id: StrictStr
    scenario_id: StrictStr
    lift_start_s: float = Field(strict=True, gt=0, allow_inf_nan=False)
    contract_sha256: StrictStr
    capture_sha256: StrictStr
    scenario_sha256: StrictStr
    native_pose_sha256: StrictStr
    seeds: dict[str, StrictInt]

    @model_validator(mode="after")
    def identities(self):
        if any(not getattr(self, name).strip() for name in ("attempt_id", "scenario_id", *IDENTITIES)):
            raise ValueError("explicit nonempty grouping identities required")
        if set(self.seeds) != {"generator", "physics", "sensor"} or any(x < 0 for x in self.seeds.values()):
            raise ValueError("generator/physics/sensor seeds required")
        if any(not _HASH.fullmatch(getattr(self, name)) for name in ("contract_sha256", "capture_sha256", "scenario_sha256", "native_pose_sha256")):
            raise ValueError("explicit source/scenario hashes required")
        return self


class Schedule(StrictModel):
    schema_version: Literal["manosim_native_task_schedule_v1"]
    domain: Literal[NATIVE_DOMAIN] = NATIVE_DOMAIN
    outcome_selection: Literal[False] = False
    source_profile_sha256: StrictStr
    hand_version_sha256: StrictStr
    label_rule_id: StrictStr
    label_rule_sha256: StrictStr
    contracts: tuple[NativeRecordingContract, ...]
    attempts: tuple[ScheduledAttempt, ...]

    @model_validator(mode="after")
    def cohort(self):
        if not self.label_rule_id.strip() or any(not _HASH.fullmatch(getattr(self, key)) for key in ("source_profile_sha256", "hand_version_sha256", "label_rule_sha256")):
            raise ValueError("native source/hand/label identities required")
        contracts = {c.contract_sha256: c for c in self.contracts}
        if not contracts or len(contracts) != len(self.contracts) or not self.attempts or len({a.attempt_id for a in self.attempts}) != len(self.attempts):
            raise ValueError("unique nonempty contracts/attempts required")
        if {a.contract_sha256 for a in self.attempts} != set(contracts): raise ValueError("exact scheduled contract coverage required")
        keys = ("assets_sha256", "shape_sha256", "collider_sha256", "controller_sha256", "controller_protocol_id", "sampling_protocol_id", "control_interval_s")
        if len({tuple(getattr(c, k) for k in keys) for c in self.contracts}) != 1:
            raise ValueError("one native hand/controller/feature cohort required; scene model hashes may differ")
        parents = {key: {} for key in IDENTITIES[1:]}
        for a in self.attempts:
            for key in parents:
                prior = IDENTITIES[:IDENTITIES.index(key)]
                value = tuple(getattr(a, k) for k in prior)
                if parents[key].setdefault(getattr(a, key), value) != value: raise ValueError("conflicting grouping identity " + key)
        captures = {}
        for a in self.attempts:
            if captures.setdefault(a.observation_group_id, a.capture_sha256) != a.capture_sha256: raise ValueError("capture hash conflict")
        return self


def freeze_splits(schedule_path):
    """Outcome-independent whole-category folds; callable before recordings exist."""
    schedule = Schedule.model_validate(read(schedule_path))
    categories = sorted({a.category_id for a in schedule.attempts}, key=lambda x: hashlib.sha256(("native_category_order_v1:" + x).encode()).hexdigest())
    if len(categories) < 3: raise ValueError("three disjoint categories required for train/validation/test")
    folds = []
    for i, test in enumerate(categories):
        validation = categories[(i + 1) % len(categories)]
        assignments = {a.attempt_id: "test" if a.category_id == test else "validation" if a.category_id == validation else "train" for a in schedule.attempts}
        folds.append(dict(fold_id="held_category_" + test, test_category=test, validation_category=validation, assignments=assignments))
    return dict(schema_version="manosim_native_category_splits_v1", schedule=binding(schedule_path), selection_uses_outcomes=False, folds=folds)


def feature_contract(spec):
    return dict(schema_version="manosim_native_pre_lift_features_v1", domain=NATIVE_DOMAIN, **spec.model_dump(), fields=list(FIELDS),
                qpos_names=list(QPOS_NAMES), qvel_names=list(QVEL_NAMES), ctrl_names=list(CTRL_NAMES), channels=169, dtype="float32",
                pose_encoding="raw wxyz sign only: first nonzero component positive per ball; no normalization",
                cutoff="prospectively scheduled lift start (must match actual first lift interval), minus lead_intervals*control_interval",
                excluded_features=["contacts", "full_state", "object_state", "outcomes", "mapping_status", "phase", "time", "seeds", "IDs", "provenance", "requested_ctrl"])


def prefix(arrays, contract, spec, *, lift_start_s):
    validate_native_observables(arrays, contract)
    phases = arrays["phase"].astype(str); times = arrays["time_s"]
    lifts = np.flatnonzero(phases == "lift")
    if not len(lifts): raise ValueError("no_lift_boundary")
    first = int(lifts[0]); dt = contract.control_interval_s
    if isinstance(lift_start_s, bool) or not np.isfinite(lift_start_s) or abs(float(arrays["action_interval_start_s"][first])-lift_start_s)>1e-9: raise ValueError("observed lift differs from prospectively scheduled boundary")
    cutoff = float(arrays["action_interval_start_s"][first] - spec.lead_intervals * dt)
    if np.any(np.isin(phases[:first], ["hold", "score"])) or np.any(phases[first:] == "close_fingers"):
        raise ValueError("invalid_phase_order")
    eligible = np.flatnonzero(times <= cutoff + 1e-9)
    if len(eligible) < spec.history_steps or abs(float(times[eligible[-1]]) - cutoff) > 1e-9:
        raise ValueError("missing_close_history")
    chosen = eligible[-spec.history_steps:]
    if not np.all(phases[chosen] == "close_fingers") or np.any(times[chosen] >= arrays["action_interval_start_s"][first]):
        raise ValueError("postcutoff_or_nonclosing_history")
    pose = arrays["hand_qpos"][chosen].copy(); quats = pose[:, 3:].reshape(-1, 16, 4)
    for row in quats:
        for q in row:
            if q[np.flatnonzero(q)[0]] < 0: q *= -1
    X = np.concatenate((pose, arrays["hand_qvel"][chosen], arrays["effective_ctrl"][chosen]), axis=1).astype(np.float32)
    if X.shape != (spec.history_steps, 169) or not np.isfinite(X).all(): raise ValueError("native feature overflow/layout")
    return X, cutoff, times[chosen].tolist()


@dataclass(frozen=True)
class NativeData:
    X: np.ndarray
    y: np.ndarray
    rows: tuple[dict, ...]
    excluded: tuple[dict, ...]
    schedule: Schedule
    bindings: dict
    readiness: dict


def _verify_frozen_task_protocol(path, freeze):
    root = Path(__file__).resolve().parents[3]; path = Path(path)
    if not __import__("re").fullmatch(r"[a-f0-9]{40}", freeze or "") or not path.is_relative_to(root): raise ValueError("exact frozen native task protocol commit required")
    blob = subprocess.check_output(["git", "-C", str(root), "show", freeze + ":" + str(path.relative_to(root))])
    if blob != path.read_bytes(): raise ValueError("native task protocol differs from frozen Git blob")


def _qualification(path, schedule, contract):
    """Explicit external task-gate interface; never a positive mapper certifier."""
    report = read(path)
    expected = dict(schema_version="manosim_native_task_qualification_v1", status="VERIFIED_COMPLETED_NATIVE_TASK_QUALIFICATION", scope="native_task_execution",
                    domain=NATIVE_DOMAIN, profile_sha256=schedule.source_profile_sha256, hand_version_sha256=schedule.hand_version_sha256,
                    recording_contract_sha256=contract.contract_sha256, controller_sha256=contract.controller_sha256,
                    collider_sha256=contract.collider_sha256, label_rule_sha256=schedule.label_rule_sha256,
                    task_execution_qualified=True, scientific_training_allowed=True, fixture_only=False)
    if any(report.get(k) != v for k, v in expected.items()): raise ValueError("qualified native task evidence required; fixtures/Shadow/PASS-only refused")
    files = {str(Path(path).resolve()): file_sha256(path)}
    for key in ("frozen_protocol", "independent_receipt"):
        b = Binding.model_validate(report.get(key)); files[str(b.verify())] = b.sha256
    _verify_frozen_task_protocol(report["frozen_protocol"]["path"], report.get("freeze_commit"))
    protocol = read(report["frozen_protocol"]["path"]); independent = read(report["independent_receipt"]["path"])
    identity = {k: expected[k] for k in ("domain", "profile_sha256", "hand_version_sha256", "recording_contract_sha256", "controller_sha256", "collider_sha256", "label_rule_sha256")}
    if any(protocol.get(k) != v for k, v in identity.items()) or protocol.get("scope") != "native_task_execution": raise ValueError("task protocol contract mismatch")
    if (independent.get("schema_version") != "manosim_native_task_independent_readback_v1" or independent.get("status") != "VERIFIED_NATIVE_TASK_QUALIFICATION"
            or independent.get("protocol_sha256") != report["frozen_protocol"]["sha256"] or any(independent.get(k) != v for k, v in identity.items())
            or independent.get("task_execution_qualified") is not True or independent.get("freeze_commit") != report.get("freeze_commit")): raise ValueError("independent native task evidence mismatch")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts: raise ValueError("task qualification artifact bindings required")
    for row in artifacts:
        b = Binding.model_validate(row)
        if str(b.verify()) in files: raise ValueError("duplicate task qualification binding")
        files[b.path] = b.sha256
    if independent.get("artifact_bindings_sha256") != canonical_sha256(artifacts): raise ValueError("independent task artifact inventory mismatch")
    return files


def load_data(collection_path, spec=PrefixSpec()):
    collection = read(collection_path)
    if set(collection) != {"schema_version", "schedule", "attempts", "qualifications"} or collection["schema_version"] != "manosim_native_collection_v1":
        raise ValueError("native collection manifest required")
    sb = Binding.model_validate(collection["schedule"]); schedule_path = sb.verify(); schedule = Schedule.model_validate(read(schedule_path))
    files = {str(Path(collection_path).resolve()): file_sha256(collection_path), str(schedule_path): sb.sha256}
    paths = collection["attempts"]
    if not isinstance(paths, dict) or set(paths) != {a.attempt_id for a in schedule.attempts}: raise ValueError("all scheduled attempts including refusals must be bound")
    contracts = {c.contract_sha256: c for c in schedule.contracts}; X, y, rows, excluded = [], [], [], []
    for a in schedule.attempts:
        ab = Binding.model_validate(paths[a.attempt_id]); attempt_json = ab.verify()
        if attempt_json.name != "attempt.json": raise ValueError("exact native attempt marker binding required")
        meta, contract, payload, arrays = load_native_attempt(attempt_json.parent)
        if meta.attempt_id != a.attempt_id or contract != contracts[a.contract_sha256]: raise ValueError("scheduled native attempt/contract mismatch")
        for key in IDENTITIES[1:]:
            if getattr(meta, key) != getattr(a, key): raise ValueError("scheduled grouping mismatch")
        expected = dict(source_profile=schedule.source_profile_sha256, hand_version=schedule.hand_version_sha256, capture=a.capture_sha256,
                        scenario=a.scenario_sha256, seeds=canonical_sha256(a.seeds), native_pose=a.native_pose_sha256)
        if any(meta.provenance_hashes.get(k) != v for k, v in expected.items()): raise ValueError("native source/capture/scenario/seeds provenance mismatch")
        files[str(attempt_json)] = ab.sha256
        for name, digest in payload["artifacts_sha256"].items(): files[str(attempt_json.parent / name)] = digest
        base = dict(attempt_id=a.attempt_id, **{k: getattr(a, k) for k in IDENTITIES})
        reason = None
        if meta.mapping.status is not MappingStatus.OK: reason = "mapping_or_setup_refusal"
        elif meta.execution_status != "completed": reason = "runtime_or_unexecuted"
        elif meta.outcome.label_rule_id != schedule.label_rule_id or meta.provenance_hashes.get("label_rule") != schedule.label_rule_sha256:
            raise ValueError("native label rule mismatch")
        elif meta.outcome.failure_type == "approach_collision": reason = "already_failed_approach_collision"
        elif not meta.outcome.success and meta.outcome.first_failure_time_s is None: reason = "unknown_failure_time"
        if reason is None and not meta.outcome.success and meta.outcome.first_failure_time_s > float(arrays["time_s"][-1])+1e-9: raise ValueError("failure timestamp outside recorded task horizon")
        if reason is None:
            try: values, cutoff, times = prefix(arrays, contract, spec, lift_start_s=a.lift_start_s)
            except ValueError as error:
                if str(error) not in {"no_lift_boundary", "missing_close_history"}: raise
                reason = str(error)
            else:
                if not meta.outcome.success and meta.outcome.first_failure_time_s <= cutoff + 1e-9: reason = "already_failed_at_or_before_cutoff"
                else:
                    X.append(values); y.append(int(meta.outcome.success)); rows.append(dict(base, cutoff_s=cutoff, observed_times_s=times, failure_type=meta.outcome.failure_type))
        if reason: excluded.append(dict(base, reason=reason))
    qualifications = collection["qualifications"]; reasons = []
    if not isinstance(qualifications, dict) or set(qualifications) != set(contracts): reasons.append("missing_exact_native_task_qualifications")
    else:
        for key, c in contracts.items():
            qb = Binding.model_validate(qualifications[key]); qp = qb.verify()
            try: files.update(_qualification(qp, schedule, c))
            except ValueError as error: reasons.append(str(error))
    readiness = dict(schema_version="manosim_native_data_readiness_v1", scheduled=len(schedule.attempts), eligible=len(rows), excluded=len(excluded),
                     class_counts={"success": sum(y), "future_failure": len(y) - sum(y)}, exclusion_counts={k: sum(r["reason"] == k for r in excluded) for k in sorted({r["reason"] for r in excluded})},
                     group_counts={k: len({r[k] for r in rows}) for k in IDENTITIES}, qualified_native_task_evidence=not reasons,
                     scientific_training_allowed=bool(rows) and not reasons, admission_reasons=reasons, adequacy_guaranteed=False)
    return NativeData(np.stack(X) if X else np.empty((0, spec.history_steps, 169), np.float32), np.asarray(y, dtype=np.int64), tuple(rows), tuple(excluded), schedule, files, readiness)


def split_indices(data, split_path):
    manifest = read(split_path); sb = Binding.model_validate(manifest.get("schedule")); schedule_path = sb.verify()
    if manifest != freeze_splits(schedule_path) or Schedule.model_validate(read(schedule_path)) != data.schedule: raise ValueError("exact frozen category splits required")
    by_id = {r["attempt_id"]: i for i, r in enumerate(data.rows)}; result = []
    for fold in manifest["folds"]:
        indices = {role: np.asarray([by_id[k] for k, assigned in fold["assignments"].items() if assigned == role and k in by_id], dtype=np.int64) for role in ROLES}
        coverage = {role: dict(samples=len(idx), successes=int(data.y[idx].sum()), failures=int(len(idx) - data.y[idx].sum()), groups={key: len({data.rows[i][key] for i in idx}) for key in IDENTITIES}) for role, idx in indices.items()}
        for key in IDENTITIES:
            groups = [{data.rows[i][key] for i in indices[role]} for role in ROLES]
            if any(groups[i] & groups[j] for i in range(3) for j in range(i)): raise ValueError("cross-partition group leak: " + key)
        usable = all(len(idx) and len(np.unique(data.y[idx])) == 2 for idx in indices.values())
        result.append(dict(fold_id=fold["fold_id"], indices=indices, coverage=coverage, status="READY" if usable else "SKIPPED_EMPTY_OR_SINGLE_CLASS_ROLE"))
    return result


def _source_files():
    root = Path(__file__).resolve().parents[3]
    return [Path(__file__).resolve(), root / "src/grasp_failure_prediction/evaluation/manosim_recording.py", root / "src/grasp_failure_prediction/evaluation/prediction_recording.py", root / "src/grasp_failure_prediction/prediction/model.py"]


def _runtime():
    import scipy, torch
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, device="cpu")


def prepare_training(collection, splits, output, prefix_spec=PrefixSpec(), training_spec=TrainingSpec()):
    data = load_data(collection, prefix_spec)
    if not data.readiness["scientific_training_allowed"]: raise ValueError("native training not admitted: " + str(data.readiness))
    folds = split_indices(data, splits); files = dict(data.bindings); files[str(Path(splits).resolve())] = file_sha256(splits)
    protocol = dict(schema_version=SCHEMA + "_protocol", scientific_domain=NATIVE_DOMAIN, collection=binding(collection), split_manifest=binding(splits),
                    prefix_spec=prefix_spec.model_dump(), training_spec=training_spec.model_dump(), input_contract=feature_contract(prefix_spec),
                    input_bindings=files, source_bindings={str(p): file_sha256(p) for p in _source_files()}, runtime=_runtime(),
                    models=list(MODELS), maximum_candidate_fits_per_fold=24, freeze_before_fitting=True, independent_statistical_unit="proposal family nested in capture/object/category; seeds are repeats",
                    fold_readiness=[{k: v for k, v in f.items() if k != "indices"} for f in folds])
    write(output, protocol)
    return protocol


def _validate_protocol(path):
    p = read(path)
    if p.get("schema_version") != SCHEMA + "_protocol" or p.get("scientific_domain") != NATIVE_DOMAIN or p.get("freeze_before_fitting") is not True: raise ValueError("native frozen training protocol required")
    if p.get("maximum_candidate_fits_per_fold") != 24 or p.get("independent_statistical_unit") != "proposal family nested in capture/object/category; seeds are repeats": raise ValueError("native fixed fitting budget/unit changed")
    spec = PrefixSpec.model_validate(p["prefix_spec"]); train_spec = TrainingSpec.model_validate(p["training_spec"])
    data = load_data(Binding.model_validate(p["collection"]).verify(), spec); folds = split_indices(data, Binding.model_validate(p["split_manifest"]).verify())
    if not data.readiness["scientific_training_allowed"] or p["input_contract"] != feature_contract(spec) or p["runtime"] != _runtime() or p["models"] != list(MODELS): raise ValueError("native admission/feature/runtime changed")
    expected = dict(data.bindings); expected[p["split_manifest"]["path"]] = p["split_manifest"]["sha256"]
    if p.get("fold_readiness") != [{k: v for k, v in f.items() if k != "indices"} for f in folds]: raise ValueError("frozen fold coverage changed")
    if expected != p["input_bindings"] or p["source_bindings"] != {str(x): file_sha256(x) for x in _source_files()}:
        raise ValueError("training source/input closure changed")
    for name, digest in {**expected, **p["source_bindings"]}.items():
        if file_sha256(name) != digest: raise ValueError("bound training input changed")
    root = Path(__file__).resolve().parents[3]; head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    for source in [Path(path).resolve(), *map(Path, p["source_bindings"])]:
        blob = subprocess.check_output(["git", "-C", str(root), "show", head + ":" + str(source.relative_to(root))])
        if blob != source.read_bytes(): raise ValueError("commit exact native training protocol/source before fitting")
    return p, data, folds, spec, train_spec, head


def _summary(X):
    return np.concatenate((X[:, -1], X.mean(axis=1), X.std(axis=1), X[:, -1] - X[:, 0]), axis=1)


@contextmanager
def _torch_scope(torch, seed):
    previous = torch.are_deterministic_algorithms_enabled(); threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed); torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
        try: yield
        finally: torch.use_deterministic_algorithms(previous); torch.set_num_threads(threads)


def _network(torch, kind):
    if kind == "summary_mlp32": return torch.nn.Sequential(torch.nn.Linear(676, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    if kind != "temporal_gru16": raise ValueError("unknown native neural model")
    class GRU(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.gru = torch.nn.GRU(169, 16, batch_first=True); self.head = torch.nn.Linear(16, 1)
        def forward(self, x): return self.head(self.gru(x)[0][:, -1])
    return GRU()


def _fit_neural(train, y, validation, vy, kind, seed, spec):
    import torch
    a = _summary(train) if kind == "summary_mlp32" else train; b = _summary(validation) if kind == "summary_mlp32" else validation
    choices = []
    for rate in spec.neural_learning_rates:
        with _torch_scope(torch, seed):
            net = _network(torch, kind); optimizer = torch.optim.Adam(net.parameters(), lr=rate, weight_decay=spec.weight_decay)
            inputs = torch.tensor(a, dtype=torch.float32); truth = torch.tensor(y[:, None], dtype=torch.float32)
            for _ in range(spec.epochs):
                optimizer.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(net(inputs), truth)
                if not torch.isfinite(loss): raise ValueError("nonfinite native neural loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()): raise ValueError("nonfinite native neural gradient")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in net.parameters()): raise ValueError("nonfinite native neural state")
            with torch.no_grad(): probability = torch.sigmoid(net(torch.tensor(b, dtype=torch.float32))).numpy().ravel().astype(float)
            choices.append((outcome_metrics(vy, probability)["log_loss"], rate, {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}))
    best = min(choices, key=lambda x: (x[0], x[1]))
    return best[2], dict(learning_rate=best[1], validation_log_loss=best[0], candidates=[dict(learning_rate=x[1], validation_log_loss=x[0]) for x in choices])


def _predict_neural(X, kind, state):
    import torch
    with _torch_scope(torch, 0):
        net = _network(torch, kind); net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()}, strict=True); net.eval()
        values = _summary(X) if kind == "summary_mlp32" else X
        with torch.no_grad(): return torch.sigmoid(net(torch.tensor(values, dtype=torch.float32))).numpy().ravel().astype(float)


def fit_fold(X, y, indices, spec, seed):
    """Shared array-only implementation; synthetic callers establish no science."""
    if X.ndim != 3 or X.shape[2] != 169 or not np.isfinite(X).all() or y.shape != (len(X),) or not np.isin(y, [0, 1]).all(): raise ValueError("native training tensor/label contract")
    if set(indices) != set(ROLES) or any(not len(indices[r]) or len(np.unique(y[indices[r]])) != 2 for r in ROLES): raise ValueError("every disjoint role must have both classes")
    all_indices = np.concatenate([indices[r] for r in ROLES])
    if len(set(all_indices.tolist())) != len(X) or sorted(all_indices.tolist()) != list(range(len(X))): raise ValueError("disjoint full fold coverage required")
    train = indices["train"]; values = X.astype(np.float64); mean = values[train].mean(axis=(0, 1)); scale = values[train].std(axis=(0, 1)); scale = np.where(scale > 1e-8, scale, 1.)
    normalized = ((values - mean) / scale).astype(np.float32)
    if not np.isfinite(normalized).all(): raise ValueError("native normalized overflow")
    result = []
    for kind in MODELS:
        state = {}; selection = {}
        if kind == "training_prevalence":
            state["prevalence"] = np.asarray(y[train].mean()); probabilities = {r: np.full(len(indices[r]), state["prevalence"].item()) for r in ("validation", "test")}
        elif kind == "summary_logistic":
            models = [fit_logistic(_summary(normalized[train]), y[train], regularization=l2, max_iterations=spec.logistic_iterations) for l2 in spec.logistic_l2]
            if any(not m.converged for m in models): raise ValueError("native logistic candidate did not converge; no outcome-guided retries")
            losses = [outcome_metrics(y[indices["validation"]], m.predict_success(_summary(normalized[indices["validation"]])))["log_loss"] for m in models]
            best = min(range(len(models)), key=lambda i: (losses[i], spec.logistic_l2[i])); model = models[best]
            state = dict(weights=model.weights, intercept=np.asarray(model.intercept), linear_mean=model.normalizer.mean, linear_scale=model.normalizer.scale, linear_active=model.normalizer.active)
            selection = dict(l2=spec.logistic_l2[best], candidates=[dict(l2=v, validation_log_loss=s) for v, s in zip(spec.logistic_l2, losses)])
            probabilities = {r: model.predict_success(_summary(normalized[indices[r]])) for r in ("validation", "test")}
        else:
            state, selection = _fit_neural(normalized[train], y[train], normalized[indices["validation"]], y[indices["validation"]], kind, seed, spec)
            probabilities = {r: _predict_neural(normalized[indices[r]], kind, state) for r in ("validation", "test")}
        result.append(dict(model=kind, state=dict(channel_mean=mean, channel_scale=scale, **state), selection=selection, probabilities=probabilities,
                           metrics={r: outcome_metrics(y[indices[r]], probabilities[r]) for r in probabilities}))
    return result


def train(protocol_path, output):
    output = Path(output)
    if output.exists(): raise FileExistsError("fresh native model output required")
    p, data, folds, prefix_spec, spec, head = _validate_protocol(protocol_path)
    initial_protocol = binding(protocol_path)
    initial_identity = (head, canonical_sha256(p), dict(data.bindings))
    output.mkdir(); write(output / "STARTED.json", dict(protocol=initial_protocol, freeze_commit=head, scope="native_task_outcome_prediction")); rows = []; failure = None
    try:
        for fold in folds:
            if fold["status"] != "READY":
                rows.append({k: v for k, v in fold.items() if k != "indices"}); continue
            for seed in spec.seeds:
                fits = fit_fold(data.X, data.y, fold["indices"], spec, seed)
                for fit in fits:
                    directory = output / (hashlib.sha256(fold["fold_id"].encode()).hexdigest()[:16] + "_" + str(seed) + "_" + fit["model"]); directory.mkdir()
                    weights = directory / "weights.npz"; np.savez_compressed(weights, **fit["state"])
                    prediction = directory / "predictions.npz"
                    prediction_arrays = {}
                    for role, probability in fit["probabilities"].items():
                        idx = fold["indices"][role]; prediction_arrays[role + "_p_success"] = probability
                        prediction_arrays[role + "_y"] = data.y[idx]
                        prediction_arrays[role + "_attempt_ids"] = np.asarray([data.rows[i]["attempt_id"] for i in idx])
                        prediction_arrays[role + "_proposal_group_ids"] = np.asarray([data.rows[i]["proposal_group_id"] for i in idx])
                    np.savez_compressed(prediction, **prediction_arrays)
                    meta = dict(schema_version=MODEL_SCHEMA, domain=NATIVE_DOMAIN, model=fit["model"], input_contract=p["input_contract"], input_contract_sha256=canonical_sha256(p["input_contract"]),
                                fold_id=fold["fold_id"], seed=seed, collection=p["collection"], recording_contracts=[c.contract_sha256 for c in data.schedule.contracts],
                                selection=fit["selection"], metrics=fit["metrics"], weights=binding(weights), predictions=binding(prediction), training_protocol=initial_protocol, scientific_task_labels=True)
                    write(directory / "model.json", meta); rows.append(meta)
        final_p, final_data, _, _, _, final_head = _validate_protocol(protocol_path)
        if (binding(protocol_path) != initial_protocol
                or (final_head, canonical_sha256(final_p), dict(final_data.bindings)) != initial_identity):
            raise ValueError("native training transaction identity changed during fitting")
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        for marker in output.rglob("model.json"):
            marker.rename(marker.with_name("model.failed_partial.json"))
    report = dict(schema_version=SCHEMA + "_report", status="HARD_STOP" if failure else "COMPLETED" if any(r.get("model") for r in rows) else "SKIPPED_NO_READY_FOLDS",
                  failure=failure, readiness=data.readiness, results=rows, threshold=.5, seed_repeats_are_not_independent_confidence_intervals=True)
    write(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("stage", choices=("readiness", "splits", "prepare", "train")); p.add_argument("--collection", type=Path); p.add_argument("--schedule", type=Path); p.add_argument("--splits", type=Path); p.add_argument("--protocol", type=Path); p.add_argument("--output", type=Path); a = p.parse_args()
    if a.stage == "readiness": print(json.dumps(load_data(a.collection).readiness, sort_keys=True))
    elif a.stage == "splits": write(a.output, freeze_splits(a.schedule))
    elif a.stage == "prepare": prepare_training(a.collection, a.splits, a.protocol)
    else: print(json.dumps(train(a.protocol, a.output), sort_keys=True))


if __name__ == "__main__": main()
