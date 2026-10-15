"""Freeze finite HUG scenario schedules before physical outcomes exist.

Axis values are explicit inputs, never estimated from outcomes. Geometry,
category, capture and proposal families remain separate identities. Quotas are
per object and refer to attempts, not independent situations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScenarioAxes:
    mass_multipliers: tuple[float, ...] = (0.75, 1.25)
    friction_coefficients: tuple[float, ...] = (0.5, 1.0)
    table_offsets_m: tuple[float, ...] = (0.0, 0.02)
    approach_modes: tuple[str, ...] = ("vertical", "grasp_derived")
    sensor_noise_std_rad: tuple[float, ...] = (0.0, 0.002)


def _hash(value) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                  allow_nan=False).encode()).hexdigest()


def sample_scenarios(objects: list[dict], *, seed: int, attempts_per_object: int,
                     generation_seeds: tuple[int, ...], axes: ScenarioAxes = ScenarioAxes()) -> dict:
    if type(seed) is not int or seed < 0 or type(attempts_per_object) is not int or attempts_per_object < 1:
        raise ValueError("declare a nonnegative integer seed and positive finite quota")
    if not objects or len({item["object_id"] for item in objects}) != len(objects):
        raise ValueError("object IDs must be nonempty and unique")
    if not generation_seeds or len(set(generation_seeds)) != len(generation_seeds) or any(type(s) is not int or s < 0 for s in generation_seeds):
        raise ValueError("generation seeds must be distinct nonnegative integers")
    for name, values in asdict(axes).items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"empty or repeated axis: {name}")
        if name != "approach_modes":
            if not all(np.isfinite(v) and (v > 0 if name in {"mass_multipliers", "friction_coefficients"} else v >= 0) for v in values):
                raise ValueError(f"invalid numeric axis: {name}")
    if set(axes.approach_modes) - {"vertical", "grasp_derived"}:
        raise ValueError("unsupported approach mode")
    rng = np.random.default_rng(seed)
    rows = []
    geometry_categories, observation_bindings = {}, {}
    for obj in sorted(objects, key=lambda x: x["object_id"]):
        if not obj.get("category_id") or not obj.get("geometry_group_id"):
            raise ValueError("each object needs declared category and geometry IDs")
        captures = obj.get("observation_group_ids", [])
        if not captures or len(set(captures)) != len(captures):
            raise ValueError("each object needs unique observation IDs")
        if geometry_categories.setdefault(obj["geometry_group_id"], obj["category_id"]) != obj["category_id"]:
            raise ValueError("one geometry cannot cross category boundaries")
        for capture in captures:
            binding = (obj["object_id"], obj["geometry_group_id"])
            if observation_bindings.setdefault(capture, binding) != binding:
                raise ValueError("one observation cannot belong to different object geometries")
        candidates = list(itertools.product(captures, generation_seeds, axes.mass_multipliers,
                          axes.friction_coefficients, axes.table_offsets_m, axes.approach_modes,
                          axes.sensor_noise_std_rad))
        if attempts_per_object > len(candidates):
            raise ValueError("quota exceeds unique declared scenarios; declare repeats explicitly")
        # Randomization is solely from the declared seed, before any outcomes.
        selected = rng.permutation(len(candidates))[:attempts_per_object]
        for index in selected:
            observation, generation, mass, friction, table, approach, noise = candidates[int(index)]
            group = _hash({"observation": observation, "generation_seed": generation,
                           "geometry": obj["geometry_group_id"]})
            row = dict(object_id=obj["object_id"], category_id=obj["category_id"],
                       geometry_group_id=obj["geometry_group_id"], observation_group_id=observation,
                       generation_seed=generation, proposal_group_id=group,
                       mass_multiplier=mass, friction_coefficient=friction,
                       table_offset_m=table, approach_mode=approach, sensor_noise_std_rad=noise,
                       sensor_noise_seed=int(rng.integers(0, 2**32)))
            rows.append(dict(scenario_id="scenario_" + _hash(row)[7:23], **row))
    if len({r["scenario_id"] for r in rows}) != len(rows):
        raise ValueError("scenario identity collision")
    return dict(schema_version="hug_scenario_schedule_v1", seed=seed,
                attempts_per_object=attempts_per_object, axes=asdict(axes), objects=objects,
                generation_seeds=list(generation_seeds), scenarios=rows,
                counts=dict(attempts=len(rows), objects=len(objects), geometries=len(geometry_categories),
                            categories=len({r["category_id"] for r in rows}),
                            observations=len({r["observation_group_id"] for r in rows}),
                            proposal_families=len({r["proposal_group_id"] for r in rows})),
                outcome_selection=False,
                scope="Declared physical variation is a new protocol, not hidden augmentation")


def freeze_category_folds(schedule: dict) -> dict:
    """All variants/captures of a held category stay outside fitting/tuning.

    The next sorted category is validation. Remaining categories are training.
    This can be frozen before outcomes; class eligibility is checked at fitting.
    """
    rows = schedule["scenarios"]
    categories = sorted({r["category_id"] for r in rows})
    if len(categories) < 3:
        raise ValueError("category folds need at least three declared categories")
    folds = []
    for i, held in enumerate(categories):
        validation = categories[(i + 1) % len(categories)]
        roles = {r["scenario_id"]: ("test" if r["category_id"] == held else
                 "validation" if r["category_id"] == validation else "train") for r in rows}
        folds.append(dict(fold_id="held_" + held, test_category=held,
                          validation_category=validation, assignments=roles))
    return dict(schema_version="hug_category_folds_v1", schedule_sha256=_hash(schedule),
                folds=folds, category_semantics="explicit supplied taxonomy; one object is not category-wide generalization",
                frozen_before_outcomes=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attempts-per-object", type=int, required=True)
    parser.add_argument("--generation-seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("freeze into a fresh directory")
    schedule = sample_scenarios(json.loads(args.objects.read_text()), seed=args.seed,
        attempts_per_object=args.attempts_per_object, generation_seeds=tuple(args.generation_seeds))
    folds = freeze_category_folds(schedule)
    args.output.mkdir(parents=True)
    for name, value in (("schedule.json", schedule), ("folds.json", folds)):
        (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(schedule["counts"]))


if __name__ == "__main__":
    main()
