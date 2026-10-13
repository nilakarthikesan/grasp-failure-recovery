"""Validate a canonical grasp proposal without running the simulator."""

from __future__ import annotations

import argparse
import json

from .artifacts import load_proposal, proposal_content_hash
from .registry import load_provider_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal")
    parser.add_argument("--expected-hash")
    parser.add_argument("--verify-native-artifacts", action="store_true")
    args = parser.parse_args()
    proposal = load_proposal(
        args.proposal,
        expected_hash=args.expected_hash,
        verify_native_artifacts=args.verify_native_artifacts,
    )
    spec = load_provider_registry().resolve(
        proposal.provider.id, proposal.provider.config_hash
    )
    if spec.embodiment_id != proposal.embodiment_id:
        raise ValueError("provider and proposal embodiments do not match")
    if tuple(proposal.compatible_execution_protocols) != tuple(
        spec.compatible_execution_protocols
    ):
        raise ValueError("provider and proposal protocol compatibility do not match")
    if proposal.approach_semantics != spec.approach_semantics:
        raise ValueError("provider and proposal approach semantics do not match")
    print(json.dumps({
        "proposal_id": proposal.proposal_id,
        "provider_id": proposal.provider.id,
        "proposal_hash": proposal_content_hash(proposal),
    }, indent=2))


if __name__ == "__main__":
    main()
