"""Create noncolliding audit output directories from the frozen protocol."""
from pathlib import Path
import gzip
import hashlib
import json

B = Path(__file__).parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    preflight = B / "candidate_validation_preflight_only.json.gz"
    report = json.loads(gzip.decompress(preflight.read_bytes()))
    assert report["preflight_only"] and len(report["cases"]) == 6
    assert report["guards"]["actual"] == 0
    assert all(c["engineering_guard_independent_audit"]["source_preflight_status"] == "PASS" for c in report["cases"])
    assert report["script_sha256"] == sha(B / "validate_candidate.py")
    assert report["protocol_sha256"] == sha(B / "protocol.json")
    assert report["candidate_sha256"] == sha(report["candidate_path"])
    rows = []
    for name, indices in (("cohort_a", [8, 12, 16]), ("cohort_b", [20, 24, 28])):
        out = B / name
        out.mkdir(exist_ok=False)
        for filename in ("protocol.json", "protected_hashes.json"):
            (out / filename).write_bytes((B / filename).read_bytes())
            assert sha(out / filename) == sha(B / filename)
        rows.append({"working_and_output_directory": str(out), "case_indices": indices, "script_path": str(B / "validate_candidate.py"), "protocol_sha256": sha(out / "protocol.json"), "candidate_path": report["candidate_path"], "candidate_sha256": report["candidate_sha256"]})
    (B / "audit_shard_plan.json").write_text(json.dumps({"schema": "hug_noncolliding_static_audit_plan_v1", "preflight_sha256": sha(preflight), "all_six_preflights_pass": True, "shards": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
