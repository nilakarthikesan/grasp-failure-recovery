"""Post-hoc, read-only analysis of saved HUG fit and audit reports.

No geometry engine, generator, simulator, optimizer or external service is
imported. Exact query identity and approximate spatial proximity are separate.
"""
import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

PARTS = ("palm", "index", "middle", "ring", "little", "thumb")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return json.load(f)


def coordinate_key(point):
    values = tuple(float(v) for v in point)
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("a finite 3D coordinate is required")
    return values


def coverage(point, selected_points):
    """No rounding, tolerance, nearest neighbor or index alias counts as exact."""
    point = coordinate_key(point)
    chosen = tuple(coordinate_key(v) for v in selected_points)
    if not chosen:
        return {"exact_encoded_coordinate_member": False,
                "nearest_selected_site_distance_mm": None}
    return {"exact_encoded_coordinate_member": point in set(chosen),
            "nearest_selected_site_distance_mm": 1000 * min(
                math.dist(point, other) for other in chosen)}


def robust_cost(residuals, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite scale required")
    if not all(math.isfinite(v) for v in residuals):
        raise ValueError("finite residuals required")
    # Literal scalar objective, matching frozen ConstrainedEvaluation cost.
    return scale * math.fsum(math.hypot(scale, v) - scale for v in residuals)


def direction_component(diag, scale):
    result = {}
    all_residuals = []
    for part in PARTS:
        item = diag["normal_diagnostics"][part]
        residuals = item.get("residual", [])
        all_residuals.extend(residuals)
        result[part] = {
            "available": item["available"],
            "bin_count": item["fit_bin_count"],
            "valid_bin_count": item.get("valid_bin_count", 0),
            "invalid_bin_count": item.get("invalid_bin_count", 0),
            "invalid_reason_counts_overlap": item.get("invalid_reason_bin_counts", {}),
            "sum_squared_residuals": math.fsum(v * v for v in residuals),
            "exact_saved_residual_soft_l1_cost": robust_cost(residuals, scale),
        }
    return {"by_part": result,
            "sum_squared_residuals": math.fsum(v * v for v in all_residuals),
            "exact_saved_residual_soft_l1_cost": robust_cost(all_residuals, scale)}


def unique_witnesses(rows):
    """Deduplicate repeated bands within ONE case and query resolution only."""
    result = {}
    for row in rows:
        key = (row["query_index"], row["part"], row["flip_type"])
        if key in result:
            old = result[key]
            if old["query_object_m"] != row["query_object_m"]:
                raise ValueError("one query index refers to different coordinates")
            old["observed_bands_mm"].append(row["band_mm"])
        else:
            item = dict(row)
            item["observed_bands_mm"] = [item.pop("band_mm")]
            result[key] = item
    return list(result.values())


def analyze(root, protocol):
    binding = protocol["input_sha256"]
    for name, expected in binding.items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if sha(path) != expected:
            raise ValueError("frozen input changed: " + name)
    base = root / protocol["source_bundle"]
    fitted = read(base / "seed_stage/candidate_targets.json.gz")
    fit_protocol = read(base / "seed_stage/protocol.json")
    plan = {row["source_inventory_index"]: row["native_sha256"]
            for row in fit_protocol["cases"]}
    cases = fitted["cases"]
    expected = protocol["source_inventory_indices"]
    if [c["source_inventory_index"] for c in cases] != expected:
        raise ValueError("six-case coverage or order changed")
    scale = fit_protocol["solver_configuration"]["f_scale"]
    statuses, coarse_totals = Counter(), Counter()
    records = []
    for case in cases:
        index = case["source_inventory_index"]
        if case["native_sha256"] != plan[index]:
            raise ValueError("native binding differs")
        cid = case["object_id"] + "_" + case["source_case"]
        audit = read(base / "validation" / (cid + "_validation.json.gz"))
        if audit["native_sha256"] != case["native_sha256"]:
            raise ValueError("audit native differs")
        coarse_cache_name = next(name for name in fit_protocol["query_cache_sha256"]
                                 if Path(name).name == cid + "_density2mm_queries.npz")
        if case["original_query_cache_sha256"] != binding[coarse_cache_name]:
            raise ValueError("fit/cache binding differs")
        with np.load(coarse_cache_name, allow_pickle=False) as cache:
            query = cache["query_object_m"].copy()
        direction = case["direction_reference_by_part"]
        surface = case["surface_reference_by_part"]
        indices = {p: set(direction[p]["query_indices"]) for p in PARTS}
        eligible = {p: set(direction[p].get("source_candidate_indices", [])) for p in PARTS}
        direction_points = {p: query[sorted(indices[p])].tolist() for p in PARTS}
        surface_indices = {p: set(surface[p]["query_indices"]) for p in PARTS}
        surface_points = {p: surface[p]["points"] for p in PARTS}
        for part in PARTS:
            if not indices[part] <= eligible[part]:
                raise ValueError("selected normal sites outside frozen stratum")
            if len(eligible[part]) != direction[part]["original_eligible_query_count"]:
                raise ValueError("eligibility cardinality mismatch")
            for ii, point in zip(surface[part]["query_indices"], surface_points[part]):
                if coordinate_key(query[ii]) != coordinate_key(point):
                    raise ValueError("surface coordinate/index mismatch")
        original = direction_component(case["baseline_fresh_diagnostics"], scale)
        selected = direction_component(case["selected_fresh_diagnostics"], scale)
        for diag, score in ((case["baseline_fresh_diagnostics"], original),
                            (case["selected_fresh_diagnostics"], selected)):
            if not math.isclose(score["sum_squared_residuals"],
                                diag["component_sse"]["facing"], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("saved facing residual/SSE mismatch")
        out = {"index": index, "case_id": cid, "native_sha256": case["native_sha256"],
               "fixed_direction_reference": {
                   p: {"available": direction[p]["available"],
                       "bin_count_including_duplicates": len(direction[p]["query_indices"]),
                       "unique_selected_query_count": len(indices[p]),
                       "original_eligible_query_count": len(eligible[p]),
                       "original_eligible_area_mm2": direction[p]["original_eligible_area_mm2"],
                       "baseline_fitting_selection": surface[p]["selection"],
                       "unique_distance_fit_site_count": len(surface_indices[p])}
                   for p in PARTS},
               "original_direction_eligibility_exclusion_counts_overlap": case["direction_reference_coverage"],
               "objective": {
                   "original_total_soft_l1_cost": case["baseline_fresh_diagnostics"]["objective"],
                   "selected_total_soft_l1_cost": case["selected_fresh_diagnostics"]["objective"],
                   "original_facing": original, "selected_facing": selected,
                   "original_all_component_sse": case["baseline_fresh_diagnostics"]["component_sse"],
                   "selected_all_component_sse": case["selected_fresh_diagnostics"]["component_sse"],
                   "distance_and_closest_vector_individual_robust_costs": "UNAVAILABLE: per-site residual arrays are not saved in candidate_targets; component SSE does not determine soft-L1 cost.",
                   "selected_invalid_bin_count": case["selected_fresh_diagnostics"]["normal_invalid_bin_count"]},
               "resolutions": []}
        for density, auditres in audit["resolutions"].items():
            same_grid = auditres["query_cache_sha256"] == case["original_query_cache_sha256"]
            if same_grid != (density == "2"):
                raise ValueError("unexpected quadrature identity")
            all_rows, orientation_counts = [], []
            for band, bandrec in auditres["orientation"]["bands"].items():
                for part in PARTS:
                    sign = bandrec["common_same_label_by_part"][part]["pointwise_sign_check"]
                    orientation_counts.append({"band_mm": int(band), "part": part,
                                               "flip_denominators": sign["flip_denominators"],
                                               "eligible_denominator": sign["eligible_outside_visual_denominator"],
                                               "numerical_deadzone": sign["native_or_mapped_dot_in_numerical_deadzone"]})
                    for flip, witnesses in sign["witnesses"].items():
                        for witness in witnesses:
                            ii = witness["query_index"]
                            point = witness["query_object_m"]
                            dc = coverage(point, direction_points[part])
                            sc = coverage(point, surface_points[part])
                            if same_grid:
                                if coordinate_key(query[ii]) != coordinate_key(point):
                                    raise ValueError("coarse witness coordinate/index mismatch")
                                if dc["exact_encoded_coordinate_member"] != (ii in indices[part]):
                                    # Coordinate duplicates can legitimately give identity without
                                    # the same query index; preserve both rather than conflate them.
                                    pass
                            all_rows.append({"query_index": ii, "query_object_m": point,
                                             "part": part, "flip_type": flip, "band_mm": int(band),
                                             "native_facing_score": witness["native_facing_compatibility"],
                                             "selected_facing_score": witness["mapped_facing_compatibility"],
                                             "same_quadrature_index_space_as_fit": same_grid,
                                             "exact_direction_fit_index_member": ii in indices[part] if same_grid else None,
                                             "original_direction_eligible_index_member": ii in eligible[part] if same_grid else None,
                                             "exact_distance_fit_index_member": ii in surface_indices[part] if same_grid else None,
                                             "direction_coordinate_coverage": dc,
                                             "distance_coordinate_coverage": sc})
            unique = unique_witnesses(all_rows)
            member_counts = {"unique_saved_witness_records": len(unique),
                             "exact_direction_fit_encoded_coordinate_members": sum(w["direction_coordinate_coverage"]["exact_encoded_coordinate_member"] for w in unique),
                             "exact_distance_fit_encoded_coordinate_members": sum(w["distance_coordinate_coverage"]["exact_encoded_coordinate_member"] for w in unique),
                             "exact_direction_fit_query_index_members": sum(w["exact_direction_fit_index_member"] for w in unique) if same_grid else None,
                             "original_direction_eligible_query_index_members": sum(w["original_direction_eligible_index_member"] for w in unique) if same_grid else None}
            if same_grid:
                coarse_totals.update({k: v for k, v in member_counts.items() if isinstance(v, int)})
            gate = auditres["frozen_acceptance_application"]
            statuses[gate["strict_relationship_status_conditional"]] += 1
            anatomy = []
            for allowance, cond in auditres["conditional_sampling"].items():
                for band, bp in cond["bands"].items():
                    for part in PARTS:
                        new = bp["independent_parts"][part]
                        old = auditres["baseline_conditional_sampling"][allowance]["bands"][band]["independent_parts"][part]
                        anatomy.append({"uniform_numeric_allowance_m": float(allowance),
                                        "band_mm": int(band), "part": part,
                                        "original_lost_lower_mm2": old["lost_native_area"]["lower_mm2"],
                                        "selected_lost_lower_mm2": new["lost_native_area"]["lower_mm2"],
                                        "original_added_lower_mm2": old["new_mapped_area"]["lower_mm2"],
                                        "selected_added_lower_mm2": new["new_mapped_area"]["lower_mm2"]})
            out["resolutions"].append({"density_mm": float(density),
                                       "query_cache_sha256": auditres["query_cache_sha256"],
                                       "strict_status": gate["strict_relationship_status_conditional"],
                                       "orientation_status": gate["pointwise_orientation_status"],
                                       "frozen_proximity_checks": gate["proximity_checks"],
                                       "all_allowance_band_part_area_bounds": anatomy,
                                       "orientation_counts_by_part_and_band": orientation_counts,
                                       "unique_saved_witness_coverage_counts": member_counts,
                                       "unique_saved_witnesses": unique})
        records.append(out)
    return {"schema": "hug_surface_objective_coverage_posthoc_v1",
            "source_bundle": protocol["source_bundle"], "case_count": len(records),
            "resolution_count": sum(len(r["resolutions"]) for r in records),
            "strict_status_counts_unchanged": dict(statuses),
            "coarse_saved_witness_membership_counts": dict(coarse_totals),
            "all_six_total_costs_decrease": all(r["objective"]["selected_total_soft_l1_cost"] < r["objective"]["original_total_soft_l1_cost"] for r in records),
            "all_six_fixed_direction_SSEs_increase": all(r["objective"]["selected_facing"]["sum_squared_residuals"] > r["objective"]["original_facing"]["sum_squared_residuals"] for r in records),
            "all_six_fixed_direction_robust_costs_increase": all(r["objective"]["selected_facing"]["exact_saved_residual_soft_l1_cost"] > r["objective"]["original_facing"]["exact_saved_residual_soft_l1_cost"] for r in records),
            "scope": protocol["scope"], "limitations": protocol["limitations"],
            "new_generator_proposals": False, "physics_steps": 0,
            "new_eval_data": False, "training": False, "records": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    protocol = read(args.protocol)
    if sha(__file__) != protocol["analysis_script_sha256"]:
        raise ValueError("analysis script changed from frozen protocol")
    report = analyze(args.root, protocol)
    report["protocol_sha256"] = sha(args.protocol)
    report["analysis_script_sha256"] = sha(__file__)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
