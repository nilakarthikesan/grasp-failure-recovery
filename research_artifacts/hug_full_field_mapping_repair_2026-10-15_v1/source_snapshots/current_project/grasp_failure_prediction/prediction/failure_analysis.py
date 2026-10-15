"""Versioned cause-to-phase reporting, separate from physical outcome scoring.

These are taxonomy groups, not measured onset timestamps. Timeouts remain
unknown. Mapping refusals and runtime errors have no physical failure phase.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from grasp_failure_prediction.evaluation.prediction_recording import load_attempt


PHYSICAL_PHASES = {"approach_collision": "pre_contact", "no_contact": "pre_contact",
                   "failed_acquisition": "acquisition", "slip_during_hold": "hold",
                   "drop": "hold", "timeout": "unknown"}


def failure_phase(failure_type: str | None, *, success: bool) -> str | None:
    if success:
        if failure_type is not None:
            raise ValueError("success cannot have a failure cause")
        return None
    if failure_type not in PHYSICAL_PHASES:
        raise ValueError("nonphysical or unknown causes cannot become physical phase labels")
    return PHYSICAL_PHASES[failure_type]


def summarize_attempts(attempt_dirs: list[str | Path]) -> dict:
    counts = Counter(); causes = Counter(); phases = Counter(); per_object = defaultdict(Counter)
    rows = []
    seen = set()
    execution_cohort = label_cohort = None
    for directory in attempt_dirs:
        metadata, payload, _ = load_attempt(Path(directory))
        if metadata.attempt_id in seen:
            raise ValueError("duplicate attempt identity")
        seen.add(metadata.attempt_id)
        current_execution = (metadata.hand_id, metadata.execution_protocol_id)
        if execution_cohort is None:
            execution_cohort = current_execution
        elif execution_cohort != current_execution:
            raise ValueError("one failure-phase report cannot mix hand/controller protocols, including null attempts")
        if metadata.outcome is not None:
            current_label = (metadata.outcome.label_rule_id, metadata.provenance_hashes["label_rule"])
            if label_cohort is None:
                label_cohort = current_label
            elif label_cohort != current_label:
                raise ValueError("one failure-phase report cannot mix completed physical label rules or hashes")
        counts["attempted"] += 1
        counts["mapped"] += metadata.mapping.status.value == "ok"
        counts["executed"] += metadata.execution_status in {"completed", "runtime_error"}
        counts["physically_labeled"] += metadata.outcome is not None
        phase = None
        if metadata.outcome is not None:
            outcome = metadata.outcome
            key = "success" if outcome.success else "physical_failure"
            counts[key] += 1; per_object[metadata.object_id][key] += 1
            phase = failure_phase(outcome.failure_type, success=outcome.success)
            if phase is not None:
                phases[phase] += 1; causes[outcome.failure_type] += 1
        else:
            counts["null_outcome"] += 1
            counts["runtime_error" if metadata.execution_status == "runtime_error" else "refused_or_unexecuted"] += 1
        rows.append(dict(attempt_id=metadata.attempt_id, object_id=metadata.object_id,
                         execution_protocol_id=metadata.execution_protocol_id,
                         mapping_status=metadata.mapping.status.value,
                         failure_phase=phase, physical_outcome=metadata.outcome.model_dump(mode="json") if metadata.outcome else None))
    cohort = dict(hand_id=None if execution_cohort is None else execution_cohort[0],
                  execution_protocol_id=None if execution_cohort is None else execution_cohort[1],
                  label_rule_id=None if label_cohort is None else label_cohort[0],
                  label_rule_sha256=None if label_cohort is None else label_cohort[1])
    return dict(schema_version="hug_failure_phase_report_v1", cohort=cohort, counts=dict(counts),
                physical_failure_causes=dict(causes), physical_failure_phases=dict(phases),
                per_object={key: dict(value) for key, value in per_object.items()}, attempts=rows,
                scope="Cause-based taxonomy; not measured slip onset or three-way prediction targets")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("phase reports are immutable")
    paths = [args.attempts.parent / name for name in json.loads(args.attempts.read_text())]
    report = summarize_attempts(paths)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
