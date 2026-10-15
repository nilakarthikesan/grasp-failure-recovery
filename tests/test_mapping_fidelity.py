"""Manufactured evidence binding only; no actual geometry or positive proof."""
import json
from pathlib import Path

import pytest

from grasp_failure_prediction.evaluation import mapping_fidelity as gate


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))


@pytest.fixture
def case(tmp_path):
    native=tmp_path/"native.pkl";native.write_bytes(b"manufactured native params, never unpickled")
    mapped=tmp_path/"mapping.json"
    write(mapped,dict(schema_version="hug_mapping_record_v1",native_proposal_sha256=gate._sha(native),
        qpos_rad=[.1,.2],world_palm_transform=[[1,0,0,.1],[0,1,0,.2],[0,0,1,.3],[0,0,0,1]]))
    roles={}
    for role in sorted(gate.GEOMETRY_ROLES):
        p=tmp_path/(role+".geometry");p.write_text("manufactured "+role)
        roles[role]={str(p):gate._sha(p)}
    rule=tmp_path/"acceptance.json";rule.write_bytes(gate.DEFAULT_ACCEPTANCE_RULE.read_bytes())
    numerical=dict(schema_version="hug_mapping_validation_v1",static_input_gate="PASS",native_surface_fidelity_accepted=False)
    arguments=dict(native_path=native,mapped_path=mapped,geometry_bindings=roles,numerical_report=numerical,acceptance_rule_path=rule)
    identities,_=gate._identities(**arguments)
    artifact=tmp_path/"supporting_receipt.json";write(artifact,dict(manufactured=True,geometry_not_measured=True))
    evidence=tmp_path/"fidelity.json"
    value=dict(schema_version=gate.EVIDENCE_SCHEMA,claim=gate.CLAIM,evidence_kind="full_domain_certificate",status="PASS",
        identities=identities,artifact_bindings={str(artifact):gate._sha(artifact)},
        coverage=dict(domain="full_actual_compiled_visual_object_triangle_domain",parts=list(gate.PARTS),bands_mm=[1,2,5],
                      continuous_proximity=True,continuous_orientation=True),certifier="arbitrary_self_declared_verifier")
    write(evidence,value)
    return dict(arguments=arguments,evidence=evidence,value=value,native=native,mapped=mapped,artifact=artifact,rule=rule)


def test_absent_evidence_refuses_before_any_file_read_or_identity_computation(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail("absent evidence attempted a read/heavy identity operation")
    monkeypatch.setattr(gate,"_sha",forbidden);monkeypatch.setattr(gate,"_identities",forbidden)
    monkeypatch.setattr(Path,"open",forbidden)
    with pytest.raises(gate.MappingFidelityRefused) as refused:
        gate.require_fidelity_pass(None,mapped_path="/does/not/exist")
    value=refused.value.verdict
    assert value["reason_codes"]==["missing_fidelity_evidence"]
    assert value["fidelity_status"]=="UNRESOLVED" and not value["collection_admitted"]
    assert value["outcome"] is None and not value["physical_outcome_label_assigned"]


def test_numerical_pass_never_implies_fidelity():
    value=gate.assess_fidelity_evidence(None,numerical_report={"static_input_gate":"PASS"})
    assert value["numerical_status"]=="PASS" and value["fidelity_status"]=="UNRESOLVED"
    assert not value["collection_admitted"]


@pytest.mark.parametrize("kind",["full_domain_certificate","full_domain_measurement","capped_witness_check","sampled_diagnostic","structural_identity_control"])
def test_no_self_declared_positive_capability_or_partial_control_admits(case,kind):
    case["value"]["evidence_kind"]=kind;write(case["evidence"],case["value"])
    value=gate.assess_fidelity_evidence(case["evidence"],**case["arguments"])
    assert value["evidence_integrity_status"]=="PASS" and value["reported_fidelity_status"]=="PASS"
    assert value["fidelity_status"]=="UNRESOLVED" and value["supported_continuous_certifiers"]==[]
    assert value["geometry_recomputed"] is False
    with pytest.raises(gate.MappingFidelityRefused):gate.require_fidelity_pass(case["evidence"],**case["arguments"])


@pytest.mark.parametrize("reported",["FAIL","UNRESOLVED"])
def test_hash_bound_negative_or_unresolved_receipts_remain_refused(case,reported):
    case["value"].update(status=reported,evidence_kind="capped_witness_check");write(case["evidence"],case["value"])
    value=gate.assess_fidelity_evidence(case["evidence"],**case["arguments"])
    assert value["evidence_integrity_status"]=="PASS" and value["fidelity_status"]==reported
    assert value["outcome"] is None and not value["collection_admitted"]
    if reported=="FAIL":assert "not replayed" in value["status_basis"]


@pytest.mark.parametrize("change",["native","mapped_root","native_hand","mapped_hand","object","acceptance","numerical","artifact"])
def test_actual_identity_or_support_mutation_fails_closed(case,change):
    if change=="native":case["native"].write_bytes(b"other native")
    elif change=="mapped_root":
        mapped=json.loads(case["mapped"].read_text());mapped["world_palm_transform"][0][3]+=.01;write(case["mapped"],mapped)
    elif change in gate.GEOMETRY_ROLES:
        path=Path(next(iter(case["arguments"]["geometry_bindings"][change])));path.write_text("changed geometry")
    elif change=="acceptance":case["rule"].write_text("changed strict rule")
    elif change=="numerical":case["arguments"]["numerical_report"]["static_input_gate"]="FAIL"
    else:case["artifact"].write_text("changed supporting receipt")
    value=gate.assess_fidelity_evidence(case["evidence"],**case["arguments"])
    assert value["evidence_integrity_status"]=="FAIL" and value["fidelity_status"]=="UNRESOLVED"
    assert not value["collection_admitted"]


def test_supplied_geometry_hash_update_does_not_rebind_old_evidence(case):
    geometry=case["arguments"]["geometry_bindings"]["mapped_hand"];path=Path(next(iter(geometry)))
    path.write_text("new geometry version");geometry[str(path)]=gate._sha(path)
    value=gate.assess_fidelity_evidence(case["evidence"],**case["arguments"])
    assert value["evidence_integrity_status"]=="FAIL" and "identity mismatch" in value["error"]["message"]


@pytest.mark.parametrize("bad",[True,False,1,{"original_index":0},{"status":"PASS"},{"path":"/missing"}])
def test_booleans_original_indexes_and_inline_pass_are_not_evidence(bad):
    value=gate.assess_fidelity_evidence(bad)
    assert value["evidence_integrity_status"]=="FAIL" and not value["collection_admitted"]


def test_hash_bound_row_reference_and_tamper(case):
    reference=dict(path=str(case["evidence"]),sha256=gate._sha(case["evidence"]))
    with pytest.raises(gate.MappingFidelityRefused) as refused:
        gate.require_row_fidelity(dict(mapping_fidelity_evidence=reference),**case["arguments"])
    assert refused.value.verdict["evidence_integrity_status"]=="PASS"
    case["evidence"].write_text(case["evidence"].read_text()+" ")
    with pytest.raises(gate.MappingFidelityRefused) as refused:
        gate.require_row_fidelity(dict(mapping_fidelity_evidence=reference),**case["arguments"])
    assert refused.value.verdict["evidence_integrity_status"]=="FAIL"
    with pytest.raises(gate.MappingFidelityRefused) as refused:
        gate.require_row_fidelity(dict(original_index=0,static_input_gate="PASS"))
    assert refused.value.verdict["reason_codes"]==["missing_fidelity_evidence"]


def test_real_structural_control_schema_does_not_transfer_to_hug(case):
    write(case["evidence"],dict(schema="paired_triangle_structural_certificate_v1",status="PASS_STRUCTURAL_IDENTITY_ON_DECLARED_DOMAIN",
        real_HUG_Shadow_acceptance=False,historical_C2_acceptance=False))
    assert gate.assess_fidelity_evidence(case["evidence"],**case["arguments"])["evidence_integrity_status"]=="FAIL"


@pytest.mark.parametrize("role",sorted(gate.GEOMETRY_ROLES))
def test_each_geometry_role_must_be_nonempty(case,role):
    case["arguments"]["geometry_bindings"][role]={}
    assert gate.assess_fidelity_evidence(case["evidence"],**case["arguments"])["evidence_integrity_status"]=="FAIL"


def test_canonical_mapped_artifact_must_retain_exact_native_provider_output(case):
    mapped=dict(schema_version="grasp_proposal_v1",embodiment_id="shadow_hand_right",provider=dict(id="hug_mano_shadow_v1"),
        native_artifacts=[dict(role="provider_output",path=case["native"].name,sha256=gate._sha(case["native"]))])
    write(case["mapped"],mapped)
    identities,_=gate._identities(**case["arguments"])
    assert identities["mapped_content_sha256"]==gate._digest(mapped)
    case["value"]["identities"]=identities;write(case["evidence"],case["value"])
    assert gate.assess_fidelity_evidence(case["evidence"],**case["arguments"])["evidence_integrity_status"]=="PASS"
    mapped["native_artifacts"][0]["path"]="../escaped.pkl";write(case["mapped"],mapped)
    assert gate.assess_fidelity_evidence(case["evidence"],**case["arguments"])["evidence_integrity_status"]=="FAIL"


@pytest.mark.parametrize("numerical,default,strict",[("PASS",0,3),("FAIL",1,1),("INCOMPLETE",2,2),(None,0,3)])
def test_static_cli_preserves_numerical_exits_unless_strict_fidelity_is_requested(numerical,default,strict):
    import importlib.util
    path=Path(__file__).parents[1]/"scripts/validate_hug_mapping.py"
    spec=importlib.util.spec_from_file_location("static_mapping_cli",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    record=dict(runtime=dict(static_input_gate=numerical) if numerical else None,
                fidelity=dict(fidelity_status="UNRESOLVED",collection_admitted=False))
    assert module._exit_code([record])==default
    assert module._exit_code([record],True)==strict
