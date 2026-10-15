"""Run a deterministic batch of canonical grasp evaluation cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .execution import ProposalRejected, run_proposal_case


def run_batch(
    case_paths: list[str | Path],
    output_root: str | Path,
    *,
    record_video: bool = True,
    verify_native_artifacts: bool = False,
    project_root: str | Path | None = None,
) -> dict:
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"batch output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    seen: set[str] = set()
    for raw_path in case_paths:
        path = Path(raw_path)
        # Parse and validate inside the runner; this read only chooses a stable
        # output name and catches duplicate case IDs before overwriting.
        payload = json.loads(path.read_text()) if path.suffix == ".json" else None
        if payload is None:
            import yaml
            payload = yaml.safe_load(path.read_text())
        case_id = payload.get("case_id") if isinstance(payload, dict) else None
        if not isinstance(case_id, str):
            raise ValueError(f"case has no string case_id: {path}")
        if case_id in seen:
            raise ValueError(f"duplicate case_id in batch: {case_id}")
        seen.add(case_id)
        try:
            result = run_proposal_case(
                path,
                root / case_id,
                record_video=record_video,
                verify_native_artifacts=verify_native_artifacts,
                project_root=project_root,
            )
            results.append({
                "case_id": result.case_id,
                "provider_id": result.provider_id,
                "success": result.success,
                "failure_type": result.failure_type,
                "status": result.status,
                "rejection_code": None,
                "result": f"{case_id}/result.json",
            })
        except ProposalRejected as exc:
            rejection = json.loads((root / case_id / "rejection.json").read_text())
            results.append({
                "case_id": case_id,
                "provider_id": rejection["provider_id"],
                "success": None,
                "failure_type": None,
                "status": exc.rejection.status,
                "rejection_code": exc.rejection.code,
                "result": f"{case_id}/rejection.json",
            })
    summary = {
        "schema_version": "eval_batch_result_v1",
        "case_count": len(results),
        "success_count": sum(item["success"] is True for item in results),
        "failure_count": sum(item["success"] is False for item in results),
        "rejection_count": sum(item["success"] is None for item in results),
        "results": results,
    }
    (root / "batch_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--no-video", dest="record_video", action="store_false")
    parser.add_argument("--verify-native-artifacts", action="store_true")
    parser.set_defaults(record_video=True)
    args = parser.parse_args()
    summary = run_batch(
        args.cases,
        args.output_root,
        record_video=args.record_video,
        verify_native_artifacts=args.verify_native_artifacts,
        project_root=args.project_root,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
