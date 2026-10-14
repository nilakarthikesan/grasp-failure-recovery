"""Provider wiring checks only: no model inference or simulation rollouts."""

import pytest
import yaml

from grasp_failure_prediction.proposals.generate import generate_proposals
from grasp_failure_prediction.proposals.providers import load_local_provider
from grasp_failure_prediction.proposals.registry import load_provider_registry


HISTORICAL_PROVIDER_HASHES = {
    "hug_mano_shadow_v1": "sha256:8da5ee897a3c65a7907da4a67b8bacabf57110f0118ffa64c3b29d2e1e8f732e",
    "shadow_saved_record_v1": "sha256:d28cf95e6a42268f57369d6c69baca5a9ce04566378029a388274cd87ff6199a",
    "dexgrasp_anything_shadow_v1": "sha256:3fe52bb649de07e16fe3598c5a951e40c7b14546ad5aaa85a1670b4efd1380ba",
    "dexgraspnet_saved_record_v1": "sha256:499dce8ed748d98b7e1fca539ddcde7a776684ae1e167410c59e54e56d10a0a2",
}
NEW_PROVIDERS = ("contactgen_mano_shadow_v1", "unidexgrasp_shadow_v1")


def test_adding_providers_preserves_historical_configuration_hashes():
    registry = load_provider_registry()
    for provider_id, expected in HISTORICAL_PROVIDER_HASHES.items():
        assert registry.config_hash(provider_id) == expected


@pytest.mark.parametrize("provider_id", NEW_PROVIDERS)
def test_new_model_provider_resolves_through_local_factory(provider_id, tmp_path):
    spec = load_provider_registry().resolve(provider_id)
    provider = load_local_provider(spec, project_root=tmp_path)
    assert f"{type(provider).__module__}:{type(provider).__name__}" == spec.adapter
    assert spec.kind == "model"
    assert spec.execution == "local_or_external"
    assert spec.compatible_execution_protocols == ("fixed_grasp_lift_v2",)
    assert spec.required_inputs == ("object_geometry", "provider_input")


@pytest.mark.parametrize("provider_id", NEW_PROVIDERS)
def test_missing_new_provider_manifest_stops_before_any_worker(
    provider_id, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("model worker must not start without its declared input")

    monkeypatch.setattr(
        "grasp_failure_prediction.proposals.generate._external_generate", forbidden
    )
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump({
        "request_id": "input_contract_only",
        "provider": {"id": provider_id},
        "object": {"id": "unit_object", "geometry_content_hash": "sha256:" + "a" * 64},
    }))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="missing required inputs.*provider_input"):
        generate_proposals(
            request, output, project_root=tmp_path, external_command="unused-worker"
        )
    assert not output.exists()
