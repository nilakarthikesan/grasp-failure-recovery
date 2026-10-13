"""Read, write, hash, and validate canonical proposal artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from grasp_failure_prediction.evaluation.schema import canonical_json

from .schema import GraspProposal, ProposalRequest


def file_sha256(path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposal_content_hash(proposal: GraspProposal) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(proposal)).hexdigest()


def load_request(path: str | Path) -> ProposalRequest:
    request_path = Path(path)
    payload = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("proposal request must contain a YAML/JSON mapping")
    return ProposalRequest.model_validate(payload)


def load_proposal(
    path: str | Path,
    *,
    expected_hash: str | None = None,
    verify_native_artifacts: bool = False,
) -> GraspProposal:
    proposal_path = Path(path)
    payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal = GraspProposal.model_validate(payload)
    actual = proposal_content_hash(proposal)
    if expected_hash is not None and actual != expected_hash:
        raise ValueError(
            f"proposal content hash mismatch: expected {expected_hash}, resolved {actual}"
        )
    if verify_native_artifacts:
        artifact_root = proposal_path.parent.resolve()
        for reference in proposal.native_artifacts:
            target = (proposal_path.parent / reference.path).resolve()
            if not target.is_relative_to(artifact_root):
                raise ValueError(f"native proposal artifact escapes its bundle: {reference.path}")
            if not target.is_file():
                raise FileNotFoundError(f"native proposal artifact is unavailable: {target}")
            if file_sha256(target) != reference.sha256:
                raise ValueError(f"native proposal artifact hash mismatch: {reference.path}")
    return proposal


def write_proposal(path: str | Path, proposal: GraspProposal) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(proposal) + b"\n")
    return proposal_content_hash(proposal)
