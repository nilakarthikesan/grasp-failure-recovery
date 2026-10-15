"""Source-only boundary for future independently qualified contact models.

The current experiment accepts only standard MuJoCo rigid contact. An exact
skin Jacobian plus penalty force law would be a DIFFERENT model/protocol; passing
the current fixtures cannot qualify that law or install a callback implicitly.
"""
from dataclasses import dataclass
from typing import Protocol
import numpy as np

STANDARD_MODEL = 'standard_mujoco_rigid_contact_mj340_v1'


@dataclass(frozen=True)
class ContactSnapshot:
    time_s: float
    qpos: np.ndarray
    qvel: np.ndarray
    control_interval_s: float


@dataclass(frozen=True)
class ContactForceResult:
    generalized_force_N_or_N_m: np.ndarray
    potential_energy_J: float
    dissipative_power_W: float
    source_identity_sha256: str


class FutureContactFactory(Protocol):
    model_id: str
    implementation_sha256: str

    def geometry_xml(self, fixture: dict) -> str: ...
    def forces(self, snapshot: ContactSnapshot) -> ContactForceResult: ...


def require_current_model(model_id):
    if model_id != STANDARD_MODEL:
        raise ValueError('Future contact factories require a new protocol; current fixtures do not qualify another law')
