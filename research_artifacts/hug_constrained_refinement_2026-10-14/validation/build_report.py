"""Write a compact evidence report after the complete frozen audit exists."""
from pathlib import Path
import hashlib
import json

B = Path(__file__).parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fmt(value, digits=3):
    return f"{value:.{digits}f}"

def main():
    s = json.loads((B / "summary.json").read_text())
    assert s["case_count"] == 6 and s["resolution_count"] == 9
    assert s["all_protected_hashes_unchanged"] and s["guards_actual_attempts"] == 0
    assert s["all_engineering_source_preflights_pass"]
    n = s["strict_status_counts"]
    lines = [
        "# HUG constrained refinement: independent static audit",
        "",
        "Date: 2026-10-14. Scope: the same six saved HUG development grasps on apple, milk box and mug, using their original native predictions, meshes, frames, commands and query cells. This artifact contains static geometry checks. It generates no evaluation episodes, generator proposals, outcomes, labels or training data.",
        "",
        f"The constrained solver returns {s['selection_source_counts']['original_fallback']} exact-original fallbacks and {s['returned_surface_correction_count']} accepted surface corrections. A lower raw fit cost does not satisfy the fingertip constraints and is not deployed. The original production mapping and its surface issue remain unchanged.",
        "",
        f"The final source preflight passes for all six cases: every fingertip error is at most its original error plus the predeclared 10 nm arithmetic allowance, and the actual mean fingertip error is at most 10 mm. The unchanged strict mapping rule returns **{n['PASS']} PASS, {n['FAIL']} FAIL and {n['UNRESOLVED']} UNRESOLVED** across six original coarse and three fine resolutions. Engineering preflight and mapping acceptance are separate claims.",
        "",
        "## Frozen method and what is independently checked",
        "",
        "The joint fit starts from each original 24-joint target, uses source URDF limits and a ±0.35 rad trust region, and fits the same preselected six-part unsigned distance and closest-point vector targets. It additionally fits the native-relative direction score `-dot(native hand outward normal, object outward normal)` on fixed original-valid area quadrature. Negative native away scores remain negative targets. Each available part has 64 deterministic area-quantile bins with mass 1/6; duplicate bins are intentional, and missing part mass is not redistributed. An invalid candidate normal comparison retains a fixed nonzero residual of 2 times the square root of its fixed weight. These fit terms are proxies, not scientific acceptance gates.",
        "",
        "The solver records major iterations and its raw final result. It selects the lowest recorded eligible objective, retaining the earliest major iterate on an exact tie; the raw final result replaces it only on strictly lower cost. The independent audit checks this selection from the recorded q, fingertip vectors and objective trace, and independently recomputes original and final FK errors. It does not recompute all trace poses or certify a global optimum. Convergence, feasibility and strict mapping validity are reported separately.",
        "",
        "The palm position stays at the native MANO wrist. Palm orientation follows `R_world_operator @ R_base_palm(q)` and can change with WRJ1/2; root translation compensates only the base-to-palm offset. Physical limits, scene precision and independent fixed-URDF versus scene FK are checked. No arbitrary hand shift or scale is introduced.",
        "",
        "The full audit reconstructs fresh compiled colliders and uses the literal prior distance, orientation eligibility, conditional cell-bound and strict-status functions. Bands remain 1, 2 and 5 mm; cell radii, source normals, visual meshes, numerical allowances, signed-facing rules and every denominator remain unchanged. Original cells are reused byte-for-byte. The additional unit-normal sanity threshold in the fitting helper is an API numerical check, not a relaxed angular or sign gate.",
        "",
        "## Case results",
        "",
    ]
    for c in s["cases"]:
        old = c["original_landmarks"]
        new = c["constrained_landmarks"]
        selection = c["selection"]
        bd = c["fit_diagnostics"]["original"]
        cd = c["fit_diagnostics"]["constrained"]
        bin0 = bd["normal_invalid_bin_count"]
        bin1 = cd["normal_invalid_bin_count"]
        iterations = selection["major_iteration"]
        selected = selection["source"] + (f" {iterations}" if iterations is not None else "")
        palm_change = c["declared_palm_geometric_change"]
        if selection["source"] == "original_fallback":
            assert all(v == 0 for v in palm_change["WRJ_delta_rad"].values())
            palm_text = "The returned exact-original pose preserves palm orientation"
        else:
            palm_text = f"Declared palm orientation changes {fmt(palm_change['rotation_angle_degrees'])}°"
        lines.append(f"- **{c['case_id']}**: mean TIP {fmt(old['mean_fingertip_error_m']*1000)} → {fmt(new['mean_fingertip_error_m']*1000)} mm; maximum TIP {fmt(old['maximum_fingertip_error_m']*1000)} → {fmt(new['maximum_fingertip_error_m']*1000)} mm; mean middle {fmt(old['mean_middle_error_m']*1000)} → {fmt(new['mean_middle_error_m']*1000)} mm. Same-stratum distance MAE {fmt(bd['fixed_quadrature_unsigned_distance_mae_m']*1000)} → {fmt(cd['fixed_quadrature_unsigned_distance_mae_m']*1000)} mm and closest-point mean error {fmt(bd['fixed_quadrature_closest_point_mean_error_m']*1000)} → {fmt(cd['fixed_quadrature_closest_point_mean_error_m']*1000)} mm. Fixed-bin invalid normals {bin0} → {bin1} of {cd['normal_bin_count']}; selected {selected}, {selection['eligible_major_count']} eligible major iterates, solver success {selection['solver_success']}, budget exhausted {selection['budget_exhausted']}. {palm_text}, with position delta {palm_change['position_delta_m']:.3g} m.")
        raw = c["raw_fit_recorded_refusal_diagnostics"]
        if raw is not None:
            lines.append(f"  Rejected raw fit: reported objective reduction {fmt(raw['fraction_objective_reduction_from_same_version_baseline'] * 100, 1)}%; largest reported TIP worsening {fmt(max(raw['tip_error_increase_m']) * 1000)} mm; {sum(not x for x in raw['allowance_individual_tip_nonworsening_pass'])}/5 individual caps fail. These are recorded callback values, not an independent full-grid audit of the rejected pose.")
    lines += [
        "",
        "All five TIP errors and all five middle errors, in thumb/index/middle/ring/little order, are retained in `summary.json`. Raw final refusal reasons, selection refusals, convergence messages and actual callback counts are preserved; no unsuccessful original proposal is removed.",
        "The unrounded rotation-angle diagnostic in `summary.json` is an arccos readout after removing the common operator serialization. Mug B has a roughly 1.7 × 10⁻⁶ degree floating-point residual even though its returned q/root and both WRJ targets are exactly original. This residual is numerical roundoff, not a physical palm-orientation change.",
        "",
        "## Full original-grid preservation measurements",
        "",
        "The comparisons below use the 2 mm proximity band on original cells with maximum triangle area 4 mm². Lost and added areas are sums of six separate part-area lower bounds; parts can overlap, so these are not unique hand area or physical contact area. Direction comparisons use each mapping's actual exterior, unique, same-part, unburied, outside-visual eligible domain; changes in raw counts alone are not evidence of improved preservation.",
        "",
    ]
    for c in s["cases"]:
        b = c["resolutions"]["2"]["bands"]["2"]
        areas = b["six_part_area_sums"]
        old = b["original_orientation_domain"]
        new = b["constrained_orientation_domain"]
        iou0 = b["original_whole_hand_unsigned_proximity"]["centroid_estimates"]["iou"]
        iou1 = b["constrained_whole_hand_unsigned_proximity"]["centroid_estimates"]["iou"]
        lines.append(f"- **{c['case_id']}**: unsigned centroid IoU {fmt(iou0)} → {fmt(iou1)}; definite LOST {fmt(areas['original_lost_lower_mm2'])} → {fmt(areas['constrained_lost_lower_mm2'])} mm²; definite ADDED {fmt(areas['original_added_lower_mm2'])} → {fmt(areas['constrained_added_lower_mm2'])} mm²; reliable direction reversals {old['sampled_sign_reversal_count']}/{old['outside_visual_eligible_denominator']['query_count']} → {new['sampled_sign_reversal_count']}/{new['outside_visual_eligible_denominator']['query_count']} eligible queries.")
    lines += ["", "### Controlling remaining discrepancies", ""]
    for c in s["cases"]:
        checks = c["resolutions"]["2"]["complete_rule_application"]["proximity_checks"]
        failing = [r for r in checks if r["status"] == "FAIL"]
        if failing:
            strongest = max(failing, key=lambda r: max(r["lost_area_lower_mm2"], r["added_area_lower_mm2"]))
            lines.append(f"- **{c['case_id']}**: largest coarse-grid definite discrepancy is {strongest['part']} at {strongest['band_mm']} mm: LOST {fmt(strongest['lost_area_lower_mm2'])} mm² and ADDED {fmt(strongest['added_area_lower_mm2'])} mm² lower bounds. This localizes a remaining surface mismatch; it does not prove that correction is globally infeasible.")
        available = []
        for band in (1, 2, 5):
            domain = c["resolutions"]["2"]["bands"][str(band)]["constrained_orientation_domain"]
            for direction, witnesses in domain["representative_reliable_signed_witnesses"].items():
                available.extend((band, direction, w) for w in witnesses)
        if available:
            band, direction, witness = max(available, key=lambda row: min(abs(row[2]["native_facing_compatibility"]), abs(row[2]["mapped_facing_compatibility"])))
            q = "[" + ", ".join(fmt(v * 1000) for v in witness["query_object_m"]) + "]"
            lines.append(f"  A retained exact-grid signed witness is {witness['native_part']} / {direction} at band {band} mm, object-local point {q} mm: native score {fmt(witness['native_facing_compatibility'], 5)}, mapped score {fmt(witness['mapped_facing_compatibility'], 5)}. Original exterior/unique/same-part/unburied/outside-visual eligibility remains satisfied at this witness.")
    lines += ["", "The complete 1/2/5 mm per-part lower/upper bounds, whole-hand intervals and candidate/valid/exterior orientation denominators are retained for all nine resolutions in the compressed independent reports and `summary.json`. Centroid IoU is an estimate; positive definite lost/added lower bounds or reliable sign reversals reject strict preservation under the original rule. An absence of sampled reversals remains UNRESOLVED for continuous orientation, not PASS.", "", "## Scope and remaining claims", "", "These are reused development cases, so fitting and checking them establishes only this scoped before/after geometry result. Full continuous physical contact equivalence remains UNRESOLVED. Failure-label preservation, unseen-object validation and predictor generalization remain NOT_TESTED. Native MANO and simulator colliders are different physical surface representations; no anatomical equivalence claim is made. A lower fit objective or passing the fingertip constraints does not establish strict mapping acceptance, successful grasping, or failure prediction accuracy.", "", f"All {s['protected_file_count']} protected source, API/test, assets, input and paper hashes match the pre-run snapshot. Actual physics integration attempts: {s['guards_actual_attempts']}. Inference, evaluation data and training: false.", "", "## Evidence bindings", "", f"- Candidate protocol SHA256: `{s['candidate_protocol_sha256']}`.", f"- Independent wrapper protocol SHA256: `{s['protocol_sha256']}`.", f"- Summary script SHA256: `{s['summary_script_sha256']}`.", "- `candidate_validation_8,12,16.json.gz` and `candidate_validation_20,24,28.json.gz`: complete independent reports.", "- `exact_direction.py`, manufactured tests and `original_direction_eligibility_parity.json`: fixed normal eligibility parity and invalid-site behavior.", "- `constrained_candidate_tradeoffs.png` and `constrained_fingertip_errors.png`: scientific diagnostic figures, bound to the summary by `figure_metadata.json`.", "- `protected_hashes.json`: the pre-run 114-file snapshot. Prior protocol reports remain separate.", ""]
    (B / "REPORT.md").write_text("\n".join(lines))
    metadata = {"schema": "hug_constrained_static_report_builder_v1", "script_sha256": sha(__file__), "summary_sha256": sha(B / "summary.json"), "report_sha256": sha(B / "REPORT.md"), "facts_derive_only_from_completed_reports": True}
    (B / "report_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"report": str(B / "REPORT.md"), "metadata": metadata}, indent=2))

if __name__ == "__main__":
    main()
