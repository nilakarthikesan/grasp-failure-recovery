"""Manufactured selector controls only; no saved grasps or geometry runs."""
import numpy as np
import pytest

from grasp_failure_prediction.evaluation.constrained_surface_refinement import ConstrainedEvaluation
from grasp_failure_prediction.evaluation.feasible_surface_refinement import FeasibleRefinementConfig, solve_feasible_refinement
from grasp_failure_prediction.evaluation.surface_component_guard import freeze_direction_component_guard


Q0 = np.zeros(24)
BOUNDS = np.tile([-1., 1.], (24, 1))
CONFIG = FeasibleRefinementConfig(maximum_pattern_sweeps=0, maximum_full_geometry_evaluations=6)


def run(guard=None, seeds=(.01, .02)):
    def tips(q):
        return np.tile([.005, 0., 0.], (5, 1))
    def full(q):
        return ConstrainedEvaluation([.03-q[1]], tips(q))
    proposals = []
    for amount in seeds:
        q = Q0.copy(); q[1] = amount; proposals.append(q)
    return solve_feasible_refinement(Q0, BOUNDS, full, evaluate_fingertips=tips,
        config=CONFIG, predeclared_seed_qpos_rad=proposals, candidate_guard=guard)


def decision(ok, *reasons):
    return {"accepted": bool(ok), "refusal_reasons": list(reasons)}


def test_guard_refusal_skips_first_seed_and_selects_later_feasible_improvement():
    seen = []
    def guard(q, value):
        assert not q.flags.writeable and not value.residuals.flags.writeable
        seen.append(q.copy())
        return decision(q[1] >= .015, *([] if q[1] >= .015 else ["manufactured_direction_regression"]))
    q, report = run(guard)
    assert q[1] == .02
    assert report['proposals'][0]['refusal'] == 'candidate_guard_refused'
    assert report['proposals'][1]['accepted']
    assert len(seen) == 3 and np.array_equal(seen[-1], q)
    assert report['events'][-1]['kind'] == 'candidate_guard_assessment'
    assert report['events'][-1]['terminal']
    assert not report['mapping_fidelity_acceptance']


def test_pattern_search_guard_refusal_continues_to_later_coordinate():
    tips = lambda q: np.tile([.005, 0., 0.], (5, 1))
    full = lambda q: ConstrainedEvaluation([.03-q[1]], tips(q))
    config = FeasibleRefinementConfig(maximum_pattern_sweeps=1, maximum_full_geometry_evaluations=8)
    guard = lambda q, value: decision(q[1] > 0, *([] if q[1] > 0 else ['direction_regression']))
    q, report = solve_feasible_refinement(Q0, BOUNDS, full, evaluate_fingertips=tips,
                                         config=config, candidate_guard=guard)
    assert q[1] == config.initial_pattern_radius_rad
    assert [row['refusal'] for row in report['proposals'][:2]] == ['candidate_guard_refused']*2
    assert report['proposals'][2]['accepted']
    assert report['events'][-1]['terminal']


def test_later_guard_refusal_preserves_prior_accepted_pose():
    q, report = run(lambda q, v: decision(q[1] < .015, *([] if q[1] < .015 else ['regression'])))
    assert q[1] == .01
    assert report['proposals'][0]['accepted'] and not report['proposals'][1]['accepted']
    assert report['surface_candidate_returned']


def test_fresh_terminal_guard_refusal_reverts_to_exact_original():
    count = 0
    def guard(q, value):
        nonlocal count
        count += 1
        return decision(count == 1, *([] if count == 1 else ['fresh_direction_regression']))
    q, report = run(guard, seeds=(.02,))
    assert np.array_equal(q, Q0)
    assert report['termination'] == 'fresh_final_verification_refusal'
    assert 'fresh final candidate guard refused' in report['callback_or_geometry_refusal']
    assert report['counters']['full'] == 4


def test_guard_exception_after_acceptance_uses_existing_safe_fallback():
    def guard(q, value):
        if q[1] > .015:
            raise RuntimeError('manufactured guard failure')
        return decision(True)
    q, report = run(guard)
    assert np.array_equal(q, Q0)
    assert report['termination'] == 'geometry_or_callback_refusal'
    assert 'manufactured guard failure' in report['callback_or_geometry_refusal']


@pytest.mark.parametrize('result', [None, {}, {'accepted': True, 'refusal_reasons': ['contradiction']},
    {'accepted': False, 'refusal_reasons': []}, {'accepted': 1, 'refusal_reasons': []},
    {'accepted': True, 'refusal_reasons': [], 'value': float('nan')}])
def test_malformed_guard_results_fail_closed(result):
    q, report = run(lambda q, v: result)
    assert np.array_equal(q, Q0)
    assert 'candidate guard callback failed' in report['callback_or_geometry_refusal']


@pytest.mark.parametrize('field', ['q', 'residuals', 'tips'])
def test_callback_cannot_mutate_solver_or_evaluation_inputs(field):
    def guard(q, value):
        array = q if field == 'q' else value.residuals if field == 'residuals' else value.fingertip_error_vectors_m
        array.setflags(write=True); array.flat[0] += .1
        return decision(True)
    q, report = run(guard)
    assert np.array_equal(q, Q0)
    assert 'modified callback inputs' in report['callback_or_geometry_refusal']


def test_guard_audit_detaches_callback_owned_results_and_absent_hook_keeps_report():
    shared = decision(True)
    _, guarded = run(lambda q, v: shared)
    shared['refusal_reasons'].append('changed_after_return')
    assert guarded['proposals'][0]['candidate_guard']['refusal_reasons'] == []
    q, legacy = run()
    assert q[1] == .02
    assert 'candidate_guard' not in legacy['counters']
    assert all('candidate_guard' not in row for row in legacy['proposals'])
    assert legacy['schema'] == 'hug_feasible_only_static_pattern_refinement_v1'


def test_invalid_guard_rejected_before_search():
    with pytest.raises(ValueError, match='candidate_guard'):
        run(guard=1)


def entry(score, valid=None):
    return {"available": True, "candidate_score": list(score),
            "candidate_valid": [True]*len(score) if valid is None else list(valid)}


def reference(native, weights=None):
    return {"available": True, "native_score": list(native),
            "weights": ([1/6/len(native)]*len(native) if native else []) if weights is None else list(weights)}


def test_identity_guard_acceptance_is_only_selector_acceptance_and_native_away_preserved():
    refs = {'thumb': reference([-.5, .5]), 'palm': {'available': False}}
    base = {'thumb': entry([-.4, .4]), 'palm': {'available': False}}
    guard = freeze_direction_component_guard(refs, base)
    identical = guard.assess(base)
    assert identical['accepted'] and not identical['scientific_mapping_acceptance']
    assert not identical['weights_renormalized'] and not identical['missing_part_mass_redistributed']
    assert guard.assess({'thumb': entry([-.5, .5]), 'palm': {'available': False}})['accepted']
    changed = guard.assess({'thumb': entry([.5, .5]), 'palm': {'available': False}})
    assert not changed['accepted'] and 'thumb:direction_sse_increased' in changed['refusal_reasons']


def test_global_sse_gain_cannot_pay_for_one_part_regression():
    refs = {'thumb': reference([0.]), 'index': reference([0.])}
    base = {'thumb': entry([1.]), 'index': entry([.1])}
    guard = freeze_direction_component_guard(refs, base)
    result = guard.assess({'thumb': entry([0.]), 'index': entry([.2])})
    assert not result['accepted']
    assert result['parts']['thumb']['direction_sse_nonincrease']
    assert not result['parts']['index']['direction_sse_nonincrease']


def test_invalid_site_cannot_disappear_even_when_part_sse_improves():
    refs = {'index': reference([0., 0.], weights=[1e-6, .1])}
    base = {'index': entry([.2, 1.])}
    result = freeze_direction_component_guard(refs, base).assess({'index': entry([0., 0.], [False, True])})
    part = result['parts']['index']
    assert not result['accepted'] and part['direction_sse_nonincrease']
    assert part['candidate_direction_sse'] == 4e-6
    assert part['candidate_invalid_fixed_mass'] == 1e-6
    assert part['original_valid_sites_invalidated_indices'] == [0]
    assert part['fixed_site_count'] == 2


@pytest.mark.parametrize('candidate', [
    {}, {'index': entry([])}, {'index': entry([float('nan')])},
    {'index': entry([float('inf')])}, {'index': entry([1.1])},
    {'index': {'available': True, 'candidate_score': [0.], 'candidate_valid': [1]}},
    {'index': {**entry([0.]), 'weights': [1.]}},
    {'index': {**entry([0.]), 'native_score': [.2]}},
    {'index': entry([0.]), 'unexpected': {'available': False}},
])
def test_malformed_missing_or_changed_candidate_evidence_refuses(candidate):
    guard = freeze_direction_component_guard({'index': reference([0.])}, {'index': entry([.1])})
    assert not guard.assess(candidate)['accepted']


def test_original_arrays_cannot_change_frozen_reference():
    refs = {'index': reference([-.5])}; base = {'index': entry([-.4])}
    guard = freeze_direction_component_guard(refs, base)
    refs['index']['native_score'][0] = .5; refs['index']['weights'][0] = 100.
    base['index']['candidate_score'][0] = .5; base['index']['candidate_valid'][0] = False
    assert guard.assess({'index': entry([-.5])})['accepted']
    assert guard.assess({'index': entry([-.5])})['parts']['index']['fixed_weight_mass'] == 1/6


def test_unavailable_parts_never_become_preservation_pass_or_receive_redistributed_mass():
    refs = {'index': reference([0.]), 'palm': {'available': False}}
    base = {'index': entry([.1]), 'palm': {'available': False}}
    guard = freeze_direction_component_guard(refs, base)
    assert not guard.assess({'index': entry([0.])})['accepted']
    assert not guard.assess({'index': entry([0.]), 'palm': entry([0.])})['accepted']
    out = guard.assess(base)
    assert out['parts']['palm']['preservation_status'] == 'UNAVAILABLE'
    assert out['parts']['index']['fixed_weight_mass'] == 1/6
    none = freeze_direction_component_guard({'palm': {'available': False}}, {'palm': {'available': False}})
    assert not none.assess({'palm': {'available': False}})['accepted']


@pytest.mark.parametrize('ref', [reference([]), reference([float('nan')]), reference([0.], [0.]),
    reference([0.], [-1.]), reference([0., .2], [.1]), reference([2.])])
def test_invalid_frozen_reference_rejected(ref):
    with pytest.raises(ValueError):
        freeze_direction_component_guard({'index': ref}, {'index': entry([0.])})
