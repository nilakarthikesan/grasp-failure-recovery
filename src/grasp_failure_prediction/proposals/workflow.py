"""High-level experiment workflow for grasp proposal generation and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator

from grasp_failure_prediction.evaluation.schema import (
    ID_PATTERN,
    EmbodimentReference,
    InitialCondition,
    ObjectCase,
    RegistryReference,
    StrictModel,
    canonical_json,
)

from .artifacts import load_proposal, load_request
from .execution import run_proposal_case
from .generate import _validate_requirements, generate_proposals
from .registry import load_provider_registry
from .schema import (
    EvaluationCaseV2,
    ProposalCaseReference,
    ProposalManifest,
)


def _relative_path(value: Path | None, field: str) -> Path | None:
    if value is not None and (value.is_absolute() or ".." in value.parts or not value.name):
        raise ValueError(f"{field} must be a relative file path")
    return value


class ExperimentEvaluation(StrictModel):
    """One physical execution of a generated proposal sample."""

    case_id: str = Field(pattern=ID_PATTERN)
    proposal_sample_index: int = Field(default=0, ge=0)
    environment: RegistryReference
    embodiment: EmbodimentReference
    execution_protocol: RegistryReference
    object: ObjectCase
    initial_condition: InitialCondition
    motion_profile_id: str = Field(pattern=ID_PATTERN)
    seed: int = Field(ge=0, le=2**32 - 1)


class ExperimentJob(StrictModel):
    """A proposal request and the evaluations that consume its samples."""

    id: str = Field(pattern=ID_PATTERN)
    request: Path
    provider_id: str | None = Field(default=None, pattern=ID_PATTERN)
    external_command: str | None = None
    endpoint: str | None = None
    bearer_token_env: str | None = None
    evaluations: tuple[ExperimentEvaluation, ...] = Field(min_length=1)

    @field_validator("request")
    @classmethod
    def relative_request(cls, value: Path) -> Path:
        result = _relative_path(value, "request")
        assert result is not None
        return result

    @model_validator(mode="after")
    def one_remote_runtime(self):
        if self.external_command is not None and self.endpoint is not None:
            raise ValueError("a job cannot set both external_command and endpoint")
        return self


class GraspEvalExperiment(StrictModel):
    """Versioned definition consumed by the ``grasp-eval`` command."""

    schema_version: Literal["grasp_eval_experiment_v1"] = "grasp_eval_experiment_v1"
    experiment_id: str = Field(pattern=ID_PATTERN)
    output_root: Path | None = None
    jobs: tuple[ExperimentJob, ...] = Field(min_length=1)
    record_video: bool = True
    verify_native_artifacts: bool = False

    @field_validator("output_root")
    @classmethod
    def relative_output(cls, value: Path | None) -> Path | None:
        return _relative_path(value, "output_root")

    @model_validator(mode="after")
    def unique_ids(self):
        job_ids = [job.id for job in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("experiment job IDs must be unique")
        case_ids = [item.case_id for job in self.jobs for item in job.evaluations]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("experiment case IDs must be unique")
        return self


def load_experiment(path: str | Path) -> GraspEvalExperiment:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment must contain a YAML/JSON mapping")
    return GraspEvalExperiment.model_validate(payload)


def _project_path(project: Path, relative: Path, label: str) -> Path:
    result = (project / relative).resolve()
    if not result.is_relative_to(project):
        raise ValueError(f"{label} escapes the project root")
    return result


def experiment_output_root(
    experiment: GraspEvalExperiment,
    project_root: str | Path,
    output_override: str | Path | None = None,
) -> Path:
    project = Path(project_root).resolve()
    if output_override is None:
        relative = experiment.output_root or Path("runs") / experiment.experiment_id
        return _project_path(project, relative, "experiment output")
    output = Path(output_override)
    result = output.resolve() if output.is_absolute() else (project / output).resolve()
    if not result.is_relative_to(project):
        raise ValueError("experiment output must remain inside the project root")
    return result


def validate_experiment(
    experiment_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> dict:
    project = Path(project_root).resolve()
    experiment = load_experiment(experiment_path)
    registry = load_provider_registry()
    job_summaries = []
    for job in experiment.jobs:
        request_path = _project_path(project, job.request, "proposal request")
        request = load_request(request_path)
        provider_id = job.provider_id or request.provider.id
        spec = registry.resolve(
            provider_id,
            request.provider.expected_config_hash if provider_id == request.provider.id else None,
        )
        if spec.embodiment_id != request.embodiment_id:
            raise ValueError(f"job {job.id}: provider embodiment does not match request")
        _validate_requirements(request, spec, project)
        for evaluation in job.evaluations:
            if evaluation.proposal_sample_index >= request.sample_count:
                raise ValueError(
                    f"job {job.id}: proposal_sample_index "
                    f"{evaluation.proposal_sample_index} exceeds request sample_count"
                )
            if evaluation.object.id != request.object.id:
                raise ValueError(
                    f"job {job.id}: evaluation object does not match proposal request"
                )
            if evaluation.embodiment.id != request.embodiment_id:
                raise ValueError(
                    f"job {job.id}: evaluation embodiment does not match proposal request"
                )
            if evaluation.execution_protocol.id not in spec.compatible_execution_protocols:
                raise ValueError(
                    f"job {job.id}: provider is incompatible with execution protocol "
                    f"{evaluation.execution_protocol.id}"
                )
        job_summaries.append(
            {
                "job_id": job.id,
                "provider_id": provider_id,
                "request_id": request.request_id,
                "proposal_count": request.sample_count,
                "case_count": len(job.evaluations),
            }
        )
    return {
        "schema_version": experiment.schema_version,
        "experiment_id": experiment.experiment_id,
        "job_count": len(experiment.jobs),
        "case_count": sum(len(job.evaluations) for job in experiment.jobs),
        "jobs": job_summaries,
        "valid": True,
    }


def _load_manifest(path: Path) -> ProposalManifest:
    return ProposalManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _archive_incomplete(path: Path) -> Path | None:
    """Preserve an interrupted output before a resumed stage retries it."""

    if not path.exists() or not any(path.iterdir()):
        return None
    index = 1
    while True:
        archived = path.with_name(f"{path.name}.incomplete-{index:02d}")
        if not archived.exists():
            path.rename(archived)
            return archived
        index += 1


def _materialize_case(
    evaluation: ExperimentEvaluation,
    manifest: ProposalManifest,
    proposal_root: Path,
    project: Path,
    case_path: Path,
) -> EvaluationCaseV2:
    entry = manifest.proposals[evaluation.proposal_sample_index]
    proposal_path = (proposal_root / entry.proposal_path).resolve()
    if not proposal_path.is_relative_to(project):
        raise ValueError("generated proposal is outside the project root")
    proposal = load_proposal(proposal_path, expected_hash=entry.proposal_hash)
    case = EvaluationCaseV2(
        case_id=evaluation.case_id,
        environment=evaluation.environment,
        embodiment=evaluation.embodiment,
        execution_protocol=evaluation.execution_protocol,
        object=evaluation.object,
        grasp=ProposalCaseReference(
            id=proposal.proposal_id,
            proposal_path=proposal_path.relative_to(project),
            expected_proposal_hash=entry.proposal_hash,
        ),
        initial_condition=evaluation.initial_condition,
        motion_profile_id=evaluation.motion_profile_id,
        seed=evaluation.seed,
    )
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_bytes(canonical_json(case) + b"\n")
    return case


def run_experiment(
    experiment_path: str | Path,
    *,
    project_root: str | Path = ".",
    output_override: str | Path | None = None,
    hug_root: str | Path | None = None,
    checkpoint: str | Path | None = None,
    record_video: bool | None = None,
    resume: bool = False,
) -> dict:
    project = Path(project_root).resolve()
    experiment = load_experiment(experiment_path)
    validate_experiment(experiment_path, project_root=project)
    output = experiment_output_root(experiment, project, output_override)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"experiment output is not empty: {output}; use 'grasp-eval resume'"
        )
    output.mkdir(parents=True, exist_ok=True)
    resolved_experiment = output / "resolved_experiment.json"
    serialized_experiment = canonical_json(experiment) + b"\n"
    if resume and resolved_experiment.is_file():
        if resolved_experiment.read_bytes() != serialized_experiment:
            raise ValueError(
                "experiment definition changed since this output was created; "
                "use a new experiment_id/output_root"
            )
    else:
        resolved_experiment.write_bytes(serialized_experiment)
    results = []
    use_video = experiment.record_video if record_video is None else record_video
    for job in experiment.jobs:
        proposal_root = output / "proposals" / job.id
        manifest_path = proposal_root / "manifest.json"
        if resume and manifest_path.is_file():
            manifest = _load_manifest(manifest_path)
        else:
            if resume:
                _archive_incomplete(proposal_root)
            manifest = generate_proposals(
                _project_path(project, job.request, "proposal request"),
                proposal_root,
                provider_id=job.provider_id,
                project_root=project,
                hug_root=hug_root,
                checkpoint=checkpoint,
                external_command=job.external_command,
                endpoint=job.endpoint,
                bearer_token_env=job.bearer_token_env,
            )
        for evaluation in job.evaluations:
            case_path = output / "cases" / f"{evaluation.case_id}.json"
            _materialize_case(
                evaluation, manifest, proposal_root, project, case_path
            )
            run_root = output / "runs" / evaluation.case_id
            result_path = run_root / "result.json"
            if resume and result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                if resume:
                    _archive_incomplete(run_root)
                model = run_proposal_case(
                    case_path,
                    run_root,
                    record_video=use_video,
                    verify_native_artifacts=experiment.verify_native_artifacts,
                    project_root=project,
                )
                result = model.model_dump(mode="json")
            results.append(
                {
                    "case_id": evaluation.case_id,
                    "provider_id": result["provider_id"],
                    "success": result["success"],
                    "failure_type": result.get("failure_type"),
                    "result": f"runs/{evaluation.case_id}/result.json",
                }
            )
    summary = {
        "schema_version": "grasp_eval_experiment_result_v1",
        "experiment_id": experiment.experiment_id,
        "case_count": len(results),
        "success_count": sum(bool(item["success"]) for item in results),
        "failure_count": sum(not bool(item["success"]) for item in results),
        "results": results,
    }
    (output / "experiment_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def experiment_status(
    experiment_path: str | Path,
    *,
    project_root: str | Path = ".",
    output_override: str | Path | None = None,
) -> dict:
    project = Path(project_root).resolve()
    experiment = load_experiment(experiment_path)
    output = experiment_output_root(experiment, project, output_override)
    expected = [item.case_id for job in experiment.jobs for item in job.evaluations]
    completed = [
        case_id for case_id in expected if (output / "runs" / case_id / "result.json").is_file()
    ]
    failed = []
    successful = []
    for case_id in completed:
        payload = json.loads((output / "runs" / case_id / "result.json").read_text())
        (successful if payload.get("success") else failed).append(case_id)
    return {
        "experiment_id": experiment.experiment_id,
        "output_root": str(output),
        "expected": len(expected),
        "completed": len(completed),
        "remaining": len(expected) - len(completed),
        "successful": len(successful),
        "failed": len(failed),
        "complete": len(completed) == len(expected),
    }
