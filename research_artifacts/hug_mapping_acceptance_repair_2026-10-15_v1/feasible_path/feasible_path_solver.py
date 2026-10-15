"""Static feasible path through saved witnesses; no scene/runtime imports.

Each accepted transition passes fresh original tip/domain gates and the fixed
pooled protection, and strictly improves the current exact rank. Local models,
finite-difference probes and rejected line-search states are never selectable.
Stopping describes a finite search, not mapping acceptance or infeasibility.
"""
from collections import OrderedDict
from pathlib import Path
import gzip
import hashlib
import json
import time
import numpy as np
from scipy.optimize import minimize
from protected_surface_policy import protected_surface_constraint_policy

DEFAULT_CONFIGURATION = {
    'maximum_raw_callback_executions_per_case': 1500,
    'maximum_exact_checks_per_case': 96,
    'maximum_analytic_tip_jacobians_per_case': 36,
    'terminal_raw_and_exact_reserve': 2,
    'initial_trust_box_scaled': .25,
    'minimum_trust_box_scaled': .0009765625,
    'dyadic_exponents': list(range(11)),
    'surface_difference_step_scaled': .0001,
    'tip_inward_target_m': 1e-6,
    'local_regularization_weight': 1e-6,
    'local_maximum_iterations': 200,
    'local_optimizer_ftol': 1e-10,
    'epigraph_slack_upper': 1000.,
    'raw_cache_size': 128,
    'wall_clock_limit_s': 600.,
}

class BudgetStop(RuntimeError): pass


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def state_key(x):
    x = np.asarray(x, np.float64)
    if x.shape != (30,) or not np.isfinite(x).all():
        raise ValueError('A complete finite 30D state is required')
    return x.tobytes()


def literal_domain(case, x):
    x = np.asarray(x, float); state_key(x)
    bounds = np.asarray(case.bounds, float)
    limits = np.asarray(case.limits, float)
    q0 = np.asarray(case.q0, float)
    aabb = np.asarray(case.domain['world_displacement_AABB_m'], float)
    if bounds.shape != (30, 2) or limits.shape != (24, 2) or q0.shape != (24,) or aabb.shape != (3, 2):
        raise ValueError('Invalid frozen domain shapes')
    if not all(np.isfinite(a).all() for a in (bounds, limits, q0, aabb)):
        raise ValueError('Nonfinite frozen domain')
    if np.any(bounds[:, 0] > bounds[:, 1]) or np.any(limits[:, 0] > limits[:, 1]) or np.any(aabb[:, 0] > aabb[:, 1]):
        raise ValueError('Reversed frozen bounds')
    q = q0 + .025 * x[:24]; shift = .005 * x[24:27]; omega = .025 * x[27:]
    return bool(np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1]) and
                np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]) and
                np.all(q >= q0 - .35) and np.all(q <= q0 + .35) and
                np.all(shift >= aabb[:, 0]) and np.all(shift <= aabb[:, 1]) and
                np.all(np.abs(omega) <= np.deg2rad(15.)))


def exact_tip_status(raw, caps):
    errors = np.asarray(raw['task_error_vectors_m'], float)
    caps = np.asarray(caps, float)
    if errors.ndim != 2 or errors.shape[0] < 5 or errors.shape[1] != 3 or caps.shape != (5,):
        raise ValueError('Five original tip error vectors/caps required')
    if not np.isfinite(errors).all() or not np.isfinite(caps).all() or np.any(caps < 0):
        raise ValueError('Invalid exact tip input')
    norms = np.sqrt(np.einsum('ij,ij->i', errors[:5], errors[:5]))
    return {'norms_m': norms, 'mean_m': float(norms.mean()),
            'passed': bool(np.all(norms <= caps) and norms.mean() <= .010),
            'individual_pass': norms <= caps, 'mean_pass': bool(norms.mean() <= .010)}


def norm_jacobian(errors, J):
    errors = np.asarray(errors, float); J = np.asarray(J, float)
    if errors.shape != (5, 3) or J.shape != (5, 3, 30) or not np.isfinite(J).all() or not np.isfinite(errors).all():
        raise ValueError('Finite virtual-tip Jacobian required')
    norms = np.sqrt(np.einsum('ij,ij->i', errors, errors))
    out = np.zeros((5, 30)); nz = norms > 0
    out[nz] = np.einsum('ij,ijk->ik', errors[nz], J[nz]) / norms[nz, None]
    return out


def exact_rank(full):
    rank = np.asarray(full['rank'], float)
    if rank.shape != (5,) or not np.isfinite(rank).all(): raise ValueError('Finite five-component exact rank required')
    return tuple(rank.tolist())


def protection_status(full, policy):
    passed = np.asarray(full['literal_surface_pass'])
    valid = np.asarray(full['exact_orientation_fields']['valid'])
    ppass = policy['protected_literal_surface_rows']; pvalid = policy['protected_orientation_eligible_rows']
    if passed.shape != ppass.shape or valid.shape != pvalid.shape or passed.dtype != bool or valid.dtype != bool:
        raise ValueError('Exact protection masks changed shape or type')
    lost_pass = np.flatnonzero(ppass & ~passed)
    lost_valid = np.flatnonzero(pvalid & ~valid)
    return {'passed': bool(not len(lost_pass) and not len(lost_valid)),
            'lost_passing_rows': lost_pass, 'lost_orientation_eligible_rows': lost_valid}



def hard_raw_status(raw, policy):
    sm = np.asarray(raw['surface_margin'], float)
    if sm.shape != policy['hard_mask'].shape or not np.isfinite(sm).all():
        raise ValueError('Finite fixed raw surface rows required')
    adjusted = sm + policy['offset']
    failures = np.flatnonzero(policy['hard_mask'] & (adjusted < 0.))
    return {'passed': bool(not len(failures)), 'failed_hard_rows': failures,
            'minimum_hard_margin': float(adjusted[policy['hard_mask']].min()) if policy['hard_mask'].any() else None}

def local_minimax_step(x, raw, surface_J, tip_J, bounds, trust, caps, policy, config):
    """QP with linear constraints only; this function never calls scene geometry."""
    x = np.asarray(x, float)
    sm = np.asarray(raw['surface_margin'], float)
    sj = np.asarray(surface_J, float)
    norms = exact_tip_status(raw, caps)['norms_m']
    nj = norm_jacobian(np.asarray(raw['task_error_vectors_m'])[:5], tip_J)
    if sj.shape != (len(sm), 30) or not np.isfinite(sj).all() or not np.isfinite(sm).all():
        raise ValueError('Finite surface local model required')
    offsets = policy['offset']; slack = policy['slack_coefficients']
    if sm.shape != offsets.shape: raise ValueError('Frozen surface row count changed')
    tip_target = np.maximum(0., np.asarray(caps) - config['tip_inward_target_m'])
    b = np.r_[(tip_target - norms)/.001,
              (.010 - config['tip_inward_target_m'] - norms.mean())/.001,
              sm + offsets]
    A = np.zeros((len(b), 31))
    A[:5, :30] = -nj/.001; A[5, :30] = -nj.mean(0)/.001
    A[6:, :30] = sj; A[6:, -1] = slack
    lo = np.maximum(bounds[:, 0] - x, -trust)
    hi = np.minimum(bounds[:, 1] - x, trust)
    t0 = max(0., float(np.max(-(sm+offsets)[slack == 1], initial=0.))) + 1e-6
    if t0 > config['epigraph_slack_upper']: raise ValueError('Frozen epigraph upper bound too small')
    z0 = np.r_[np.zeros(30), t0]
    weight = config['local_regularization_weight']
    def fun(z): return float(z[-1] + .5*weight*np.dot(z[:30]/trust, z[:30]/trust))
    def jac(z): return np.r_[weight*z[:30]/trust**2, 1.]
    result = minimize(fun, z0, jac=jac, method='SLSQP',
                      bounds=[tuple(v) for v in np.c_[lo, hi]] + [(0., config['epigraph_slack_upper'])],
                      constraints=[{'type':'ineq', 'fun':lambda z: b+A@z, 'jac':lambda z:A}],
                      options={'maxiter':config['local_maximum_iterations'], 'ftol':config['local_optimizer_ftol'], 'disp':False})
    z = np.asarray(result.x, float)
    # A malformed numerical return cannot enter the path or produce NaN JSON.
    # Preserve its representation as text and keep the prior accepted state.
    if z.shape != (31,) or not np.isfinite(z).all() or not np.isfinite(result.fun):
        return {'step_scaled': np.zeros(30), 'epigraph': None,
                'usable_finite_trust_step': False, 'status': int(result.status),
                'success': False, 'message': 'nonfinite_or_malformed_local_optimizer_return',
                'iterations': int(result.nit), 'objective': None,
                'linearized_constraint_min': None, 'linearized_constraints': np.empty(0),
                'slack_column': A[:, -1], 'raw_numerical_return_repr': repr(result),
                'optimizer_success_not_acceptance': True}
    usable = bool(np.all(z[:30] >= lo) and np.all(z[:30] <= hi) and
                  z[-1] >= 0. and z[-1] <= config['epigraph_slack_upper'])
    return {'step_scaled': z[:30], 'epigraph': float(z[-1]), 'usable_finite_trust_step': usable,
            'status': int(result.status), 'success': bool(result.success), 'message': str(result.message),
            'iterations': int(result.nit), 'objective': float(result.fun),
            'linearized_constraint_min': float(np.min(b+A@z)),
            'linearized_constraints': b+A@z, 'slack_column': A[:, -1],
            'optimizer_success_not_acceptance': True}


def solve(case, start_state, pooled_reference_exact, configuration, out):
    config = dict(configuration)
    if set(config) != set(DEFAULT_CONFIGURATION): raise ValueError('Explicit complete frozen configuration required')
    if config != DEFAULT_CONFIGURATION: raise ValueError('This implementation supports only its declared initial pilot budget')
    x = np.asarray(start_state, float).copy(); state_key(x)
    if not literal_domain(case, x): raise ValueError('Start outside original domain')
    policy = protected_surface_constraint_policy(pooled_reference_exact)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tracepath = out/f'case_{case.index}_trace.jsonl.gz'
    if tracepath.exists(): raise FileExistsError('Never overwrite study evidence')
    stream = gzip.open(tracepath, 'wt', encoding='utf-8')
    start_clock = time.perf_counter(); count0 = (case.raw_calls, case.exact_calls, case.tip_jacobian_calls)
    events = 0; hits = 0; cache = OrderedDict(); accepted = []; models = []; current = None; current_event = None
    def emit(kind, **kw):
        nonlocal events
        events += 1
        stream.write(json.dumps(plain({'event':events, 'kind':kind, **kw}), allow_nan=False, separators=(',',':'))+'\n'); stream.flush()
        return events
    def limit(kind, reserved=True):
        reserve = config['terminal_raw_and_exact_reserve'] if reserved else 0
        if reserved and time.perf_counter()-start_clock >= config['wall_clock_limit_s']: raise BudgetStop('wall_clock_budget')
        if kind in ('raw', 'exact') and case.raw_calls-count0[0] >= config['maximum_raw_callback_executions_per_case']-reserve: raise BudgetStop('raw_callback_budget')
        if kind == 'exact' and case.exact_calls-count0[1] >= config['maximum_exact_checks_per_case']-reserve: raise BudgetStop('exact_check_budget')
        if kind == 'jac' and case.tip_jacobian_calls-count0[2] >= config['maximum_analytic_tip_jacobians_per_case']: raise BudgetStop('analytic_jacobian_budget')
    def raw(state):
        nonlocal hits
        key = state_key(state)
        if key in cache:
            hits += 1; emit('raw_cache_hit', state_scaled=state, source_event=cache[key][1])
            return cache[key][0]
        limit('raw'); value = case.raw(state)
        if not np.isfinite(np.asarray(value['surface_margin'], float)).all(): raise ValueError('Nonfinite raw margins')
        ev = emit('raw_callback', state_scaled=state, value=value)
        cache[key] = (value, ev)
        if len(cache) > config['raw_cache_size']: cache.popitem(last=False)
        return value
    def exact(state, reason, reserved=True):
        limit('exact', reserved); full = case.exact(state)
        tips = exact_tip_status(full, case.caps)
        prot = protection_status(full, policy); rank = exact_rank(full); hard = hard_raw_status(full, policy)
        domain = literal_domain(case, state) and bool(full['in_domain'])
        ev = emit('exact_checkpoint', reason=reason, value=full, independent_tip_status=tips,
                  fixed_pooled_protection=prot, hard_raw_status=hard, literal_domain_pass=domain)
        return full, ev, bool(tips['passed'] and domain and prot['passed'] and hard['passed']), rank
    try:
        current, current_event, ok, rank = exact(x, 'fresh_segment_start')
        if not ok: raise ValueError('Saved accepted start failed fresh exact gates')
        start_exact = current; start_event = current_event
        accepted.append({'event':current_event, 'state_scaled':x.copy(), 'rank':rank, 'role':'fresh_start'})
        emit('fixed_policy', policy=policy, configuration=config)
        trust = config['initial_trust_box_scaled']; stop = None
        while stop is None:
            try:
                limit('jac'); base = raw(x)
                tip_jac = case.tip_jacobian(x)
                sj = np.empty((len(base['surface_margin']), 30)); spans = []
                for k in range(30):
                    forward = min(case.bounds[k, 1], x[k]+config['surface_difference_step_scaled'])-x[k]
                    backward = x[k]-max(case.bounds[k, 0], x[k]-config['surface_difference_step_scaled'])
                    span = forward if forward >= backward else -backward
                    if span == 0: raise ValueError('Frozen domain has no finite difference span')
                    probe = x.copy(); probe[k] += span
                    sj[:, k] = (np.asarray(raw(probe)['surface_margin'])-base['surface_margin'])/span
                    spans.append(span)
                emit('local_jacobians', accepted_source_event=current_event, state_scaled=x, surface_jacobian=sj,
                     analytic_tip_jacobian=tip_jac, difference_spans_scaled=spans, trust_box_scaled=trust)
                model = local_minimax_step(x, base, sj, tip_jac, np.asarray(case.bounds), trust, case.caps, policy, config)
                model_event = emit('local_model', accepted_source_event=current_event, trust_box_scaled=trust, result=model)
                models.append({'event':model_event, 'accepted_source_event':current_event, 'trust_box_scaled':trust,
                               'status':model['status'], 'success':model['success'], 'linearized_constraint_min':model['linearized_constraint_min']})
                progressed = False
                if model['usable_finite_trust_step'] and np.any(model['step_scaled'] != 0.):
                    for exponent in config['dyadic_exponents']:
                        alpha = 2.**(-exponent); proposal = x + alpha*model['step_scaled']
                        if not literal_domain(case, proposal):
                            emit('line_search_refusal', local_model_event=model_event, exponent=exponent,
                                 state_scaled=proposal, reason='literal_domain'); continue
                        actual = raw(proposal); tips = exact_tip_status(actual, case.caps)
                        if not tips['passed']:
                            emit('line_search_refusal', local_model_event=model_event, exponent=exponent,
                                 state_scaled=proposal, reason='original_tip_caps_or_mean', tip_status=tips); continue
                        hard = hard_raw_status(actual, policy)
                        if not hard['passed']:
                            emit('line_search_refusal', local_model_event=model_event, exponent=exponent,
                                 state_scaled=proposal, reason='actual_buffered_hard_surface', hard_raw_status=hard); continue
                        full, ev, gates, proposed_rank = exact(proposal, f'local_model_{model_event}_dyadic_{exponent}')
                        strict = proposed_rank < rank
                        if not gates or not strict:
                            emit('line_search_refusal', local_model_event=model_event, exponent=exponent,
                                 source_exact_event=ev, reason='exact_gates' if not gates else 'no_strict_current_rank_improvement',
                                 current_rank=rank, proposed_rank=proposed_rank); continue
                        before = rank; x = proposal.copy(); current = full; current_event = ev; rank = proposed_rank
                        change = next(k for k, (a,b) in enumerate(zip(before, rank)) if a != b)
                        accepted.append({'event':ev, 'state_scaled':x.copy(), 'rank':rank, 'previous_rank':before,
                                         'first_improved_rank_component':change, 'local_model_event':model_event,
                                         'dyadic_exponent':exponent, 'trust_box_scaled':trust, 'role':'accepted_transition'})
                        emit('accepted_transition', accepted_exact_event=ev, transition={k:v for k,v in accepted[-1].items() if k != 'event'}); progressed = True; break
                if not progressed:
                    if trust <= config['minimum_trust_box_scaled']:
                        stop = 'no_accepted_dyadic_step_at_minimum_trust_box'
                    else:
                        old = trust; trust = max(config['minimum_trust_box_scaled'], trust/2.)
                        emit('trust_box_shrink', previous_trust_scaled=old, next_trust_scaled=trust,
                             unchanged_accepted_event=current_event)
            except BudgetStop as exc:
                stop = str(exc)
        emit('search_stop', reason=stop, last_accepted_event=current_event, not_an_infeasibility_claim=True)
        fresh, final_event, gates, final_rank = exact(x, 'fresh_returned_accepted_state', reserved=False)
        same = gates and final_rank == rank and np.array_equal(fresh['literal_surface_pass'], current['literal_surface_pass']) and np.array_equal(fresh['exact_orientation_fields']['valid'], current['exact_orientation_fields']['valid'])
        refused_terminal = None
        if not same:
            refused_terminal = {'state_scaled':x.copy(), 'accepted_event':current_event, 'fresh_event':final_event}
            emit('fresh_accepted_state_mismatch', **refused_terminal)
            x = np.asarray(start_state, float).copy()
            fresh, final_event, gates, final_rank = exact(x, 'fresh_initial_segment_fallback', reserved=False)
            if not gates or final_rank != exact_rank(start_exact) or not np.array_equal(fresh['literal_surface_pass'], start_exact['literal_surface_pass']) or not np.array_equal(fresh['exact_orientation_fields']['valid'], start_exact['exact_orientation_fields']['valid']):
                raise ValueError('Fresh initial fallback inconsistent; no candidate may be returned')
            current_event = start_event
        counts = {'raw_callback_executions':case.raw_calls-count0[0], 'exact_checks':case.exact_calls-count0[1],
                  'analytic_tip_jacobians':case.tip_jacobian_calls-count0[2], 'cache_hits':hits,
                  'local_models':len(models), 'accepted_transitions':len(accepted)-1,
                  'raw_budget_includes_exact_internal_raw':True,
                  'analytic_jacobian_staging_forward_calls_counted_separately_by_driver':True}
        assert counts['raw_callback_executions'] <= config['maximum_raw_callback_executions_per_case']
        assert counts['exact_checks'] <= config['maximum_exact_checks_per_case']
        assert counts['analytic_tip_jacobians'] <= config['maximum_analytic_tip_jacobians_per_case']
        stream.close()
        return x, fresh, {'schema':'hug_feasible_path_solver_v1', 'configuration':config, 'fixed_pooled_policy':policy,
                          'start_exact':start_exact, 'accepted_path':accepted, 'local_model_summaries':models,
                          'termination':stop, 'selected_source_exact_event':current_event, 'fresh_return_event':final_event,
                          'refused_terminal_state':refused_terminal, 'counts':counts,
                          'trace_path':str(tracepath), 'trace_sha256':hashlib.sha256(tracepath.read_bytes()).hexdigest(),
                          'runtime_s':time.perf_counter()-start_clock, 'stopping_does_not_prove_infeasibility':True,
                          'retained_witnesses_not_full_surface_validation':True}
    finally:
        if not stream.closed: stream.close()
