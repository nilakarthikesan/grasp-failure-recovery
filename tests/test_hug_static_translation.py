"""Manufactured static-profile selection/status controls; no optimizer or HUG."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from grasp_failure_prediction.evaluation.retargeting import (
    LEGACY_RETARGETING_PROFILE, STATIC_RETARGETING_PROFILE, RetargetingError, ShadowHandRetargeter,
)
from grasp_failure_prediction.proposals import translation
from test_retargeting import synthetic_hug_prediction


def strategy(profile=STATIC_RETARGETING_PROFILE):
    legacy = translation.load_translation_registry().resolve(translation.LEGACY_MANO_STRATEGY)
    payload = legacy.model_dump()
    if profile == STATIC_RETARGETING_PROFILE:
        payload.update(strategy_id=translation.STATIC_MANO_STRATEGY,
            retargeter_id="dex_retargeting_vector_shadow_right_static_v2", retargeting_profile=profile)
    return translation.TranslationStrategySpec.model_validate(payload)


def test_registered_static_strategy_is_explicit_and_legacy_registration_is_unchanged():
    registry = translation.load_translation_registry()
    legacy = registry.resolve(translation.LEGACY_MANO_STRATEGY)
    static = registry.resolve(translation.STATIC_MANO_STRATEGY)
    assert legacy.retargeting_profile in (None, LEGACY_RETARGETING_PROFILE)
    assert legacy.retargeter_id == "dex_retargeting_vector_shadow_right"
    assert static.retargeting_profile == STATIC_RETARGETING_PROFILE
    assert static.retargeter_id == "dex_retargeting_vector_shadow_right_static_v2"
    assert static.strategy_kind == "official_retargeter"
    assert static.target_asset_hash == legacy.target_asset_hash
    assert static.source_root_link == legacy.source_root_link == "wrist"
    assert static.target_root_link == legacy.target_root_link == "palm"


@pytest.mark.parametrize("profile", [LEGACY_RETARGETING_PROFILE, STATIC_RETARGETING_PROFILE])
def test_explicit_registry_profile_selection_requires_live_statuses(tmp_path, monkeypatch, profile):
    spec = strategy(profile)
    config = tmp_path / "existing_config.yaml"; config.write_text("manufactured pinned config")
    digest = "sha256:" + hashlib.sha256(config.read_bytes()).hexdigest()
    spec = spec.model_copy(update={"evidence":tuple(item.model_copy(update={"sha256":digest})
        if item.role == "retargeter_config" else item for item in spec.evidence)})
    monkeypatch.setattr(translation, "load_translation_registry", lambda: SimpleNamespace(resolve=lambda _:spec))
    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(config_path=config)
    monkeypatch.setattr(translation, "ShadowHandRetargeter", construct)
    result = translation._official_mano_retargeter(spec.strategy_id)
    assert result.config_path == config
    assert calls == [dict(profile=profile, require_optimizer_status=True)]


@pytest.mark.parametrize("changes", [
    dict(strategy_id=translation.LEGACY_MANO_STRATEGY),
    dict(strategy_id="unversioned_static_alias"),
    dict(retargeting_profile=None),
    dict(retargeting_profile=LEGACY_RETARGETING_PROFILE),
    dict(retargeter_id="dex_retargeting_vector_shadow_right"),
])
def test_static_profile_cannot_silently_replace_or_alias_legacy(changes):
    payload = strategy().model_dump(); payload.update(changes)
    with pytest.raises(ValueError):
        translation.TranslationStrategySpec.model_validate(payload)


def test_kinematic_strategy_cannot_claim_an_optimizer_profile():
    payload = translation.load_translation_registry().resolve("dex_urdf_shadow_identity_v1").model_dump()
    payload["retargeting_profile"] = STATIC_RETARGETING_PROFILE
    with pytest.raises(ValueError, match="only to official"):
        translation.TranslationStrategySpec.model_validate(payload)


@pytest.mark.parametrize("profile", [LEGACY_RETARGETING_PROFILE, STATIC_RETARGETING_PROFILE])
@pytest.mark.parametrize("recorded", [False, True])
def test_every_fresh_translation_entry_refuses_finite_optimizer_fallback(tmp_path, monkeypatch, profile, recorded):
    # Actual project status/refusal code, manufactured SeqRetargeting only.
    config = tmp_path / "config.yml";config.write_text("manufactured")
    class Sequence:
        joint_names = ("j0", "j1")
        joint_limits = np.array([[-1.,1.],[-1.,1.]])
        filter = None
        def __init__(self):
            self.calls = 0
            self.optimizer = SimpleNamespace(scaling=1., norm_delta=0. if profile == STATIC_RETARGETING_PROFILE else .004,
                target_link_human_indices=np.array([[0]*10,[4,8,12,16,20,2,6,10,14,18]]),
                opt=SimpleNamespace(last_optimize_result=lambda:-4))
        def reset(self):pass
        def retarget(self, _):
            self.calls += 1
            return np.array([.2,.3])
    target = object.__new__(ShadowHandRetargeter)
    target.profile = profile;target.require_optimizer_status = True
    target.config_path = config;target.config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    target.optimizer_iterations = 20;target._retargeting = Sequence()
    monkeypatch.setattr(translation, "_official_mano_retargeter", lambda _:target)
    grasp, _ = synthetic_hug_prediction()
    if recorded:
        pose, record = translation.translate_mano_with_mapping_record(grasp, "manufactured_strategy")
        assert pose is None and record["mapping_status"] == "refused_invalid_retarget"
        assert record["outcome"] is None
        assert record["raw_refinement_qpos_rad"] == [[.2,.3]]
        assert record["optimizer_status_codes"] == [-4]
    else:
        with pytest.raises(RetargetingError, match="termination status -4"):
            translation.translate_mano_with_official_retargeter(grasp, "manufactured_strategy")
    assert target._retargeting.calls == 1


def test_recorded_entry_rejects_unknown_positive_status_without_fabricating_outcome(monkeypatch):
    from grasp_failure_prediction.evaluation import fresh_mapping_qualification as capture
    monkeypatch.setattr(translation,"_official_mano_retargeter",lambda _:SimpleNamespace(
        joint_limits=np.array([[-1.,1.]]), optimizer_iterations=20))
    monkeypatch.setattr(capture,"capture_solve",lambda *args:(object(),dict(status="returned", optimizer_status_codes=[999]*20)))
    pose, record = translation.translate_mano_with_mapping_record(None, "manufactured")
    assert pose is None and record["optimizer_status_gate_passed"] is False
    assert record["mapping_status"] == "refused_invalid_retarget" and record["outcome"] is None
