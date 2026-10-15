"""Fail-closed admission of HUG-to-Shadow static fidelity evidence.

Numerical conversion and geometric fidelity are different claims. This module
checks immutable evidence identities; it does not solve, measure geometry,
certify surfaces, run inference or simulate. No continuous real-MANO-to-Shadow
positive certifier is implemented here. Consequently NO current evidence can
admit physical collection: an unsupported positive assertion is UNRESOLVED.
A future positive capability requires reviewed code that replays its continuous
proof on the exact native/mapped/object/hand identities, not an allow flag.

Evidence schema ``hug_shadow_mapping_fidelity_evidence_v1`` binds ``identities``
to native file, mapped file/content, named geometry-file closure, numerical
report and the unchanged acceptance rule. ``artifact_bindings`` must name every
supporting receipt by absolute path and SHA-256. A bound negative receipt may
report FAIL without implying this gate recomputed geometry. Capped witnesses,
finite samples and manufactured structural identity controls never prove PASS.
Missing evidence is refused before ANY file read or project/heavy import.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from collections.abc import Mapping

SCHEMA = "hug_mapping_fidelity_gate_v1"
EVIDENCE_SCHEMA = "hug_shadow_mapping_fidelity_evidence_v1"
CLAIM = "STRICT_SURFACE_RELATIONSHIP_PRESERVATION_AT_DECLARED_BANDS"
ACCEPTANCE_SHA256 = "sha256:87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4"
DEFAULT_ACCEPTANCE_RULE = Path(__file__).resolve().parents[3] / "research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_rule.json"
PARTS = ("palm", "thumb", "index", "middle", "ring", "little")
GEOMETRY_ROLES = frozenset({"native_hand", "mapped_hand", "object"})
KINDS = frozenset({"full_domain_certificate", "full_domain_measurement", "capped_witness_check",
                   "sampled_diagnostic", "structural_identity_control"})
SUPPORTED_CONTINUOUS_CERTIFIERS = ()  # Explicitly none for real HUG -> Shadow.


class MappingFidelityRefused(ValueError):
    """Unproved mapping admission, never a completed physical failure label."""
    def __init__(self, verdict):
        self.verdict = verdict
        super().__init__("HUG mapping fidelity is not admitted: " + ", ".join(verdict["reason_codes"]))


def _sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _tag(value):
    if not isinstance(value, str) or re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", value) is None:
        raise ValueError("a literal SHA-256 is required")
    return "sha256:" + value.removeprefix("sha256:")


def _checked_files(bindings):
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("nonempty bound file closure required")
    result = {}
    for name, expected in bindings.items():
        path = Path(name)
        if not path.is_absolute() or str(path.resolve()) in result:
            raise ValueError("bound file paths must be absolute and unique")
        actual = _sha(path)
        if actual != _tag(expected):
            raise ValueError("bound evidence/geometry file changed")
        result[str(path.resolve())] = actual
    return result


def _identities(native_path, mapped_path, geometry_bindings, acceptance_rule_path, numerical_report):
    if native_path is None or mapped_path is None:
        raise ValueError("exact native and mapped artifact paths are required with evidence")
    native, mapped = Path(native_path).resolve(), Path(mapped_path).resolve()
    if not isinstance(geometry_bindings, Mapping) or set(geometry_bindings) != GEOMETRY_ROLES:
        raise ValueError("native_hand/mapped_hand/object geometry closures are required")
    geometry = {role: _checked_files(files) for role, files in geometry_bindings.items()}
    rule = Path(acceptance_rule_path or DEFAULT_ACCEPTANCE_RULE).resolve()
    if _sha(rule) != ACCEPTANCE_SHA256:
        raise ValueError("acceptance rule differs from frozen strict HUG rule")
    native_hash, mapped_hash = _sha(native), _sha(mapped)
    content, schema = None, "binary_saved_command"
    if mapped.suffix.lower() == ".json":
        value = json.loads(mapped.read_text())
        if not isinstance(value, dict):
            raise ValueError("mapped JSON artifact must be an object")
        content, schema = _digest(value), value.get("schema_version")
        if schema == "grasp_proposal_v1":
            if (value.get("embodiment_id") != "shadow_hand_right"
                    or not value.get("provider", {}).get("id", "").startswith("hug")):
                raise ValueError("canonical mapped artifact is not a HUG Shadow proposal")
            native_refs = value.get("native_artifacts", [])
            outputs = []
            for ref in native_refs:
                target = (mapped.parent / ref["path"]).resolve()
                if not target.is_relative_to(mapped.parent) or _sha(target) != _tag(ref["sha256"]):
                    raise ValueError("canonical native artifact binding changed or escaped bundle")
                if ref["role"] == "provider_output": outputs.append(_tag(ref["sha256"]))
            if outputs != [native_hash]:
                raise ValueError("canonical mapped artifact does not identify this native proposal")
        elif schema == "hug_mapping_record_v1":
            if _tag(value.get("native_proposal_sha256")) != native_hash:
                raise ValueError("mapped record identifies another native proposal")
    numerical = dict(numerical_report) if isinstance(numerical_report, Mapping) else None
    return dict(native_sha256=native_hash, mapped_artifact_sha256=mapped_hash,
                mapped_content_sha256=content, mapped_artifact_schema=schema,
                geometry_binding_sha256=_digest(geometry), acceptance_rule_sha256=ACCEPTANCE_SHA256,
                numerical_report_sha256=_digest(numerical)), geometry


def _numerical_status(report):
    status = report.get("static_input_gate") if isinstance(report, Mapping) else None
    return status if status in {"PASS", "FAIL"} else "UNRESOLVED"


def assess_fidelity_evidence(evidence=None, *, native_path=None, mapped_path=None, geometry_bindings=None,
                             numerical_report=None, acceptance_rule_path=None):
    """Read/hash evidence only. ``evidence=None`` performs no file reads.

    Evidence can be a JSON path or a row reference ``{path, sha256}``. A full
    inline report, booleans and original-case indexes are not evidence refs.
    ``geometry_bindings`` is ``{native_hand: {absolute_path: sha}, ...}`` with
    exactly the three GEOMETRY_ROLES and no empty role. These hashes identify
    inputs; they cannot establish that the caller supplied complete geometry.
    That completeness is a future continuous verifier's obligation as well.
    """
    verdict = dict(schema_version=SCHEMA, numerical_status=_numerical_status(numerical_report),
        fidelity_status="UNRESOLVED", reported_fidelity_status=None,
        evidence_integrity_status="NOT_CHECKED", collection_admitted=False,
        reason_codes=[], supported_continuous_certifiers=list(SUPPORTED_CONTINUOUS_CERTIFIERS),
        acceptance_rule_sha256=ACCEPTANCE_SHA256, evidence_sha256=None, identities=None,
        outcome=None, physical_outcome_label_assigned=False, geometry_recomputed=False)
    if evidence is None:
        verdict["reason_codes"] = ["missing_fidelity_evidence"]
        return verdict
    try:
        expected = None
        if isinstance(evidence, Mapping):
            if set(evidence) != {"path", "sha256"}:
                raise ValueError("evidence reference must contain exactly path and sha256")
            path, expected = Path(evidence["path"]), _tag(evidence["sha256"])
        elif isinstance(evidence, (str, Path)):
            path = Path(evidence)
        else:
            raise ValueError("an evidence JSON path/reference is required")
        actual = _sha(path)
        if expected is not None and actual != expected:
            raise ValueError("fidelity evidence reference hash changed")
        value = json.loads(path.read_text())
        if (not isinstance(value, dict) or value.get("schema_version") != EVIDENCE_SCHEMA
                or value.get("claim") != CLAIM or value.get("evidence_kind") not in KINDS
                or value.get("status") not in {"PASS", "FAIL", "UNRESOLVED"}):
            raise ValueError("unsupported fidelity evidence schema/claim/status")
        identities, geometry = _identities(native_path, mapped_path, geometry_bindings, acceptance_rule_path, numerical_report)
        if value.get("identities") != identities:
            raise ValueError("native/mapped/geometry/acceptance/numerical identity mismatch")
        artifacts = _checked_files(value.get("artifact_bindings"))
        verdict.update(evidence_integrity_status="PASS", reported_fidelity_status=value["status"],
                       evidence_sha256=actual, identities=identities, geometry_bindings=geometry, artifact_bindings=artifacts)
        kind = value["evidence_kind"]
        if kind == "structural_identity_control":
            verdict["reason_codes"].append("manufactured_structural_control_is_not_real_hug_shadow_fidelity")
        elif value["status"] == "FAIL":
            verdict.update(fidelity_status="FAIL", status_basis="hash-bound negative audit receipt; geometry is not replayed by this admission gate")
            verdict["reason_codes"].append("bound_fidelity_failure_report")
        elif value["status"] == "UNRESOLVED":
            verdict["reason_codes"].append("bound_fidelity_unresolved_report")
        else:
            coverage = value.get("coverage", {})
            if (kind != "full_domain_certificate" or coverage.get("domain") != "full_actual_compiled_visual_object_triangle_domain"
                    or coverage.get("parts") != list(PARTS) or coverage.get("bands_mm") != [1, 2, 5]
                    or coverage.get("continuous_proximity") is not True or coverage.get("continuous_orientation") is not True):
                verdict["reason_codes"].append("partial_sample_or_witness_evidence_cannot_certify_full_fidelity")
            # Coverage statements and an arbitrary JSON verifier name do not
            # create a capability. No real HUG->Shadow positive verifier exists.
            verdict["reason_codes"].append("no_supported_real_hug_shadow_continuous_certifier")
        if verdict["numerical_status"] != "PASS":
            verdict["reason_codes"].append("numerical_conversion_not_accepted")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        verdict.update(evidence_integrity_status="FAIL", fidelity_status="UNRESOLVED",
                       reason_codes=["invalid_or_mismatched_fidelity_evidence"], error=dict(type=type(error).__name__, message=str(error)))
    return verdict


def require_fidelity_pass(evidence=None, **identity_inputs):
    """Refuse unless a supported verifier established PASS (none exist today)."""
    verdict = assess_fidelity_evidence(evidence, **identity_inputs)
    if verdict["fidelity_status"] != "PASS" or verdict["numerical_status"] != "PASS" or not verdict["collection_admitted"]:
        raise MappingFidelityRefused(verdict)
    return verdict


def require_row_fidelity(proposal_row, **identity_inputs):
    """Collection integration: no evidence ref is an immediate null-label refusal."""
    if not isinstance(proposal_row, Mapping):
        return require_fidelity_pass(None, **identity_inputs)
    return require_fidelity_pass(proposal_row.get("mapping_fidelity_evidence"), **identity_inputs)
