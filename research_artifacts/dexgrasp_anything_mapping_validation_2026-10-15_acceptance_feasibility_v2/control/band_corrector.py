"""Development-only proximity-band proposals with unchanged adoption guards.

Finite fitting-query improvement is never a surface preservation certificate.
The caller freezes source arrays, all inputs, current guards and implementation.
"""
import time
import numpy as np
from scipy.optimize import minimize

BANDS_M = np.array([.0005, .001, .002])
MARGIN_M = .00025
VARIABLE_SCALES = np.array([.1] * 22 + [1.] * 3 + [.02] * 3)


def band_violations(source, target):
    source, target = np.asarray(source), np.asarray(target)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 8:
        raise ValueError('Complete corresponding Nx8 fields required')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Finite fields required')
    s, t = source[:, :7, None], target[:, :7, None]
    # A small fitting margin discourages overfitting a single query boundary;
    # it changes neither the three scientific bands nor any acceptance rule.
    margin = np.minimum(np.abs(s - BANDS_M), MARGIN_M)
    visual = np.where(s <= BANDS_M, t - BANDS_M + margin,
                      BANDS_M - t + margin)
    ss, tt = source[:, 7], target[:, 7]
    margin = np.minimum(np.abs(ss), MARGIN_M)
    occupancy = np.where(ss > 0, margin - tt,
                         np.where(ss < 0, tt + margin, np.abs(tt)))
    return np.maximum(np.r_[visual.ravel(), occupancy], 0.)


def propose(ctx, core, initial, source, caps, *, deadline, max_states=2400):
    """Three fixed starts; simultaneous 28-parameter SLSQP proposals only.

    Every adoption passes old strict target/wrist/FK/landmark checks and all
    18 componentwise nonregression guards. Infeasible SQP trials are recorded
    only as workspaces, never exported as accepted mappings.
    """
    bounds = ctx.variable_bounds(caps)
    origin = initial['variables'].copy()
    scales = VARIABLE_SCALES
    current_metrics = initial['component_metrics']
    if not core.accepted_feasible(initial, caps, current_metrics):
        raise ValueError('Initial state must satisfy all unchanged guards')
    scaled_bounds = (bounds - origin[:, None]) / scales[:, None]
    native_seed = origin.copy()
    # This is an explicitly labelled search seed, not alteration of native data
    # or a final angle-clipped mapping.
    native_seed[:22] = np.clip([ctx.qsource[n] for n in ctx.fingers],
                              bounds[:22, 0], bounds[:22, 1])
    rng = np.random.default_rng(2026101502)
    random_seed = np.clip(origin + scales * rng.uniform(-.15, .15, 28),
                          bounds[:, 0], bounds[:, 1])
    starts = [('saved_component_correction', origin),
              ('native_projected_search_seed', native_seed),
              ('fixed_seed_joint_palm_perturbation', random_seed)]
    cache = {origin.tobytes(): initial}
    logs, attempts, eligible = [], [], [initial]
    per_start_limit = max_states // len(starts)
    baseline_proxy = float(np.sum((band_violations(source, initial['fields']) / .001)**2))

    class Budget(Exception):
        pass

    for label, seed in starts:
        start_count = len(logs)
        def lookup(y):
            if time.monotonic() > deadline:
                raise Budget('WALL_BUDGET')
            x = origin + np.asarray(y) * scales
            key = x.tobytes()
            if key in cache:
                return cache[key]
            if len(logs) - start_count >= per_start_limit or len(logs) >= max_states:
                raise Budget('FIELD_STATE_BUDGET')
            if not np.isfinite(x).all() or np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
                return None
            state = ctx.make_state(x, initial, 'palm')
            if not core.domain_valid(state):
                return None
            state['fields'] = ctx.evaluate_fields(state)
            state['component_metrics'] = core.component_metrics(source, state['fields'])
            state['proxy_objective'] = float(np.sum((band_violations(source, state['fields']) / .001)**2))
            state['field_archive_key'] = 'joint_band_trial_' + str(len(logs))
            feasible = core.accepted_feasible(state, caps, current_metrics)
            if feasible and core.strict_pareto_improvement(state['component_metrics'], current_metrics):
                eligible.append(state)
            logs.append({'start': label, 'scope': 'PROPOSAL_WORKSPACE',
                         'true_all_guards_pass': feasible,
                         'state': core.state_summary(state),
                         'proxy_objective': state['proxy_objective']})
            cache[key] = state
            return state

        def objective(y):
            state = lookup(y)
            return 1e12 if state is None else float(np.sum((band_violations(source, state['fields']) / .001)**2))

        def constraints(y):
            state = lookup(y)
            return (np.full(48, -1.) if state is None else
                    core.normalized_inequalities(state, current_metrics, caps, len(source)))

        try:
            result = minimize(objective, (seed - origin) / scales, method='SLSQP',
                              bounds=[tuple(x) for x in scaled_bounds],
                              constraints=[{'type': 'ineq', 'fun': constraints}],
                              options={'maxiter': 60, 'ftol': 1e-10, 'eps': 1e-4, 'disp': False})
            lookup(result.x)
            status = {'status': 'COMPLETE', 'success': bool(result.success),
                      'message': str(result.message), 'iterations': int(result.nit)}
        except Budget as error:
            status = {'status': str(error)}
        attempts.append({'start': label, 'new_states': len(logs) - start_count, **status})
    # Scientific adoption is NOT selected by the proxy. Preserve original
    # component priority and require the original strict per-component guards.
    chosen = core.select_strict_candidate(initial, eligible[1:], caps)
    if not core.accepted_feasible(chosen, caps, current_metrics):
        raise ValueError('Final true guard failure')
    return chosen, {'attempts': attempts, 'trial_count': len(logs),
                    'eligible_improved_count': len(eligible)-1,
                    'adopted_changed_state': chosen is not initial,
                    'baseline_proxy': baseline_proxy,
                    'chosen_proxy': float(np.sum((band_violations(source, chosen['fields'])/.001)**2)),
                    'acceptance_from_proxy': False,
                    'scientific_acceptance': 'NOT_TESTED',
                    'execution_eligible': False, 'trials': logs}
