from __future__ import annotations

import pytest

from grasp_failure_prediction.evaluation.schema import canonical_sha256
from grasp_failure_prediction.proposals.registry import load_provider_registry
from grasp_failure_prediction.proposals.schema import RigidTransform


def test_rigid_transform_requires_a_unit_quaternion():
    with pytest.raises(ValueError, match="unit length"):
        RigidTransform(translation_m=(0, 0, 0), quaternion_xyzw=(0, 0, 0, 2))


def test_provider_registry_hash_is_stable():
    registry = load_provider_registry()
    spec = registry.resolve("hug_mano_shadow_v1")
    assert registry.config_hash(spec.provider_id) == canonical_sha256(spec)
    assert spec.compatible_execution_protocols == ("fixed_grasp_lift_v2",)
