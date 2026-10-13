"""Generate canonical grasp proposals with local, process, or HTTP providers."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from urllib.request import Request, urlopen

from grasp_failure_prediction.evaluation.schema import canonical_json, canonical_sha256

from .artifacts import (
    file_sha256,
    load_proposal,
    load_request,
    proposal_content_hash,
    write_proposal,
)
from .providers import load_local_provider
from .registry import ProviderSpec, load_provider_registry
from .schema import (
    GraspProposal,
    ProposalManifest,
    ProposalManifestEntry,
    ProposalRequest,
    ProviderReference,
)


def _project_input(project_root: Path, relative: Path, role: str) -> Path:
    path = (project_root / relative).resolve()
    if not path.is_relative_to(project_root):
        raise ValueError(f"{role} path escapes the project root")
    return path


def _expected_input_binding(request: ProposalRequest, project_root: Path) -> dict:
    expected_manifest = None
    expected_contract = None
    if request.observation_path is not None:
        observation = _project_input(project_root, request.observation_path, "observation")
        manifest = observation / "observation_manifest.json"
        contract = observation / "scene_contract.json"
        expected_manifest = file_sha256(manifest) if manifest.is_file() else None
        expected_contract = file_sha256(contract) if contract.is_file() else None
    elif request.provider_input_path is not None:
        provider_input = _project_input(
            project_root, request.provider_input_path, "provider input"
        )
        expected_manifest = file_sha256(provider_input) if provider_input.is_file() else None
    elif request.provider_inputs:
        input_files = []
        for item in request.provider_inputs:
            path = _project_input(project_root, item.path, item.role)
            if not path.is_file():
                raise FileNotFoundError(f"provider input does not exist: {path}")
            input_files.append({
                "role": item.role,
                "path": item.path.as_posix(),
                "sha256": file_sha256(path),
            })
        expected_manifest = canonical_sha256({"inputs": input_files})
    observation_identity = {
        "input_manifest_hash": expected_manifest,
        "scene_contract_hash": expected_contract,
        "object_geometry": request.object.geometry_content_hash,
    }
    return {
        "geometry_group_id": canonical_sha256(
            {"geometry_content_hash": request.object.geometry_content_hash}
        ),
        "observation_group_id": (
            canonical_sha256(observation_identity)
            if expected_manifest is not None or expected_contract is not None
            else None
        ),
        "generation_request_group_id": canonical_sha256(request),
        "input_manifest_hash": expected_manifest,
        "scene_contract_hash": expected_contract,
    }


def _validate_requirements(
    request: ProposalRequest, spec: ProviderSpec, project_root: Path
) -> None:
    available = {item.role for item in request.provider_inputs}
    available.add("object_geometry")
    if request.provider_input_path is not None:
        available.add("saved_shadow_record")
    if request.observation_path is not None:
        observation = _project_input(project_root, request.observation_path, "observation")
        observation_files = {
            "rgb": "rgb.png",
            "depth_m": "depth_m.npy",
            "intrinsics": "intrinsics.npy",
            "object_mask": "object_mask.png",
            "scene_contract": "scene_contract.json",
            "selection_uv": "observation_manifest.json",
        }
        for role, filename in observation_files.items():
            if (observation / filename).is_file():
                available.add(role)
    missing = sorted(set(spec.required_inputs) - available)
    if missing:
        raise ValueError(
            f"provider {spec.provider_id!r} is missing required inputs: {missing}"
        )


def _empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_returned_proposal(
    path: Path,
    request: ProposalRequest,
    spec: ProviderSpec,
    *,
    sample_index: int,
    project_root: Path,
) -> GraspProposal:
    proposal = load_proposal(path)
    expected_config_hash = canonical_sha256(spec)
    if proposal.provider.id != spec.provider_id:
        raise ValueError("returned proposal provider ID does not match the request")
    if proposal.provider.config_hash != expected_config_hash:
        raise ValueError("returned proposal provider configuration hash is stale")
    if proposal.object.id != request.object.id:
        raise ValueError("returned proposal object ID does not match the request")
    if proposal.object.geometry_content_hash != request.object.geometry_content_hash:
        raise ValueError("returned proposal geometry hash does not match the request")
    if proposal.embodiment_id != request.embodiment_id:
        raise ValueError("returned proposal embodiment does not match the request")
    if proposal.generation.sample_index != sample_index:
        raise ValueError("returned proposal sample index does not match its output slot")
    if proposal.generation.seed != request.seed + sample_index:
        raise ValueError("returned proposal generation seed does not match the request")
    expected_binding = _expected_input_binding(request, project_root)
    if proposal.input_binding.geometry_group_id != expected_binding["geometry_group_id"]:
        raise ValueError("returned proposal geometry group is incorrect")
    if proposal.input_binding.generation_request_group_id != expected_binding["generation_request_group_id"]:
        raise ValueError("returned proposal request group is incorrect")
    if proposal.input_binding.input_manifest_hash != expected_binding["input_manifest_hash"]:
        raise ValueError("returned proposal input-manifest hash is incorrect")
    if proposal.input_binding.scene_contract_hash != expected_binding["scene_contract_hash"]:
        raise ValueError("returned proposal scene-contract hash is incorrect")
    if proposal.input_binding.observation_group_id != expected_binding["observation_group_id"]:
        raise ValueError("returned proposal observation group is incorrect")
    if tuple(proposal.compatible_execution_protocols) != tuple(
        spec.compatible_execution_protocols
    ):
        raise ValueError("returned proposal protocol compatibility is incorrect")
    if proposal.approach_semantics != spec.approach_semantics:
        raise ValueError("returned proposal approach semantics are incorrect")
    return proposal


def _external_generate(
    command: str,
    request_path: Path,
    output: Path,
) -> list[Path]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("--external-command must not be empty")
    subprocess.run(
        [*argv, "--request", str(request_path.resolve()), "--output", str(output.resolve())],
        check=True,
    )
    return sorted(output.glob("sample_*/proposal.json"))


def _endpoint_generate(
    endpoint: str,
    request: ProposalRequest,
    spec: ProviderSpec,
    output: Path,
    bearer_token_env: str | None,
    project_root: Path,
) -> list[Path]:
    inputs = []
    for role, relative in (
        ("observation", request.observation_path),
        ("provider_input", request.provider_input_path),
        *((item.role, item.path) for item in request.provider_inputs),
    ):
        if relative is None:
            continue
        source = _project_input(project_root, relative, role)
        files = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in files:
            if not path.is_file():
                continue
            payload = path.read_bytes()
            inputs.append({
                "role": role,
                "path": (
                    path.relative_to(source).as_posix()
                    if source.is_dir()
                    else relative.as_posix()
                ),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "content_base64": base64.b64encode(payload).decode("ascii"),
            })
    body = canonical_json(
        {
            "request": request.model_dump(mode="json"),
            "provider_spec": spec.model_dump(mode="json"),
            "provider_config_hash": canonical_sha256(spec),
            "expected_input_binding": _expected_input_binding(request, project_root),
            "inputs": inputs,
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if bearer_token_env:
        token = os.environ.get(bearer_token_env)
        if not token:
            raise ValueError(f"token environment variable is unset: {bearer_token_env}")
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(
        Request(endpoint, data=body, headers=headers, method="POST"), timeout=120
    ) as response:
        payload = json.loads(response.read())
    raw_proposals = payload.get("proposals") if isinstance(payload, dict) else None
    if not isinstance(raw_proposals, list):
        raise ValueError("provider endpoint must return {'proposals': [...]} JSON")
    paths: list[Path] = []
    for index, raw in enumerate(raw_proposals):
        proposal = GraspProposal.model_validate(raw)
        path = output / f"sample_{index:04d}" / "proposal.json"
        write_proposal(path, proposal)
        paths.append(path)
    return paths


def generate_proposals(
    request_path: str | Path,
    output_dir: str | Path,
    *,
    provider_id: str | None = None,
    project_root: str | Path | None = None,
    hug_root: str | Path | None = None,
    checkpoint: str | Path | None = None,
    external_command: str | None = None,
    endpoint: str | None = None,
    bearer_token_env: str | None = None,
) -> ProposalManifest:
    project = Path(project_root or ".").resolve()
    request_file = Path(request_path).resolve()
    request = load_request(request_file)
    provider_overridden = provider_id is not None and provider_id != request.provider.id
    if provider_overridden:
        request = request.model_copy(
            update={"provider": ProviderReference(id=provider_id)}
        )
    effective_request = request_file
    registry = load_provider_registry()
    spec = registry.resolve(request.provider.id, request.provider.expected_config_hash)
    if spec.embodiment_id != request.embodiment_id:
        raise ValueError("provider embodiment does not match proposal request")
    _validate_requirements(request, spec, project)
    selected = sum(option is not None for option in (external_command, endpoint))
    if selected > 1:
        raise ValueError("choose only one of --external-command and --endpoint")
    output = Path(output_dir)
    _empty_output(output)
    if external_command is not None:
        if provider_overridden:
            effective_request = output / "resolved_request.json"
            effective_request.write_bytes(canonical_json(request) + b"\n")
        paths = _external_generate(external_command, effective_request, output)
    elif endpoint is not None:
        paths = _endpoint_generate(
            endpoint, request, spec, output, bearer_token_env, project
        )
    else:
        provider = load_local_provider(
            spec,
            project_root=project,
            hug_root=Path(hug_root).resolve() if hug_root else None,
            checkpoint=Path(checkpoint).resolve() if checkpoint else None,
        )
        paths = provider.generate(request, output)
    if len(paths) != request.sample_count:
        raise ValueError(
            f"provider returned {len(paths)} proposals; expected {request.sample_count}"
        )
    entries = []
    seen_ids: set[str] = set()
    for sample_index, path in enumerate(paths):
        proposal = _validate_returned_proposal(
            path,
            request,
            spec,
            sample_index=sample_index,
            project_root=project,
        )
        if proposal.proposal_id in seen_ids:
            raise ValueError(f"provider returned duplicate proposal ID: {proposal.proposal_id}")
        seen_ids.add(proposal.proposal_id)
        entries.append(
            ProposalManifestEntry(
                proposal_id=proposal.proposal_id,
                proposal_path=path.resolve().relative_to(output.resolve()),
                proposal_hash=proposal_content_hash(proposal),
                generation_seed=proposal.generation.seed,
                proposal_sample_index=proposal.generation.sample_index,
                geometry_group_id=proposal.input_binding.geometry_group_id,
                observation_group_id=proposal.input_binding.observation_group_id,
                generation_request_group_id=(
                    proposal.input_binding.generation_request_group_id
                ),
            )
        )
    manifest = ProposalManifest(
        request_id=request.request_id,
        provider_id=spec.provider_id,
        provider_config_hash=canonical_sha256(spec),
        proposals=tuple(entries),
    )
    (output / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("--provider", help="registered provider ID; overrides the request")
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--hug-root")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--external-command",
        help="command that accepts --request PATH --output DIRECTORY",
    )
    parser.add_argument("--endpoint", help="HTTP endpoint returning canonical proposals")
    parser.add_argument("--bearer-token-env")
    args = parser.parse_args()
    manifest = generate_proposals(
        args.request,
        args.output,
        provider_id=args.provider,
        project_root=args.project_root,
        hug_root=args.hug_root,
        checkpoint=args.checkpoint,
        external_command=args.external_command,
        endpoint=args.endpoint,
        bearer_token_env=args.bearer_token_env,
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
