"""Static practical DGA retargeter; no execution or native-fidelity claim.

The unique Euclidean box projection is an explicit reference mapping, never a
hidden alteration of source data. Optional corrections operate only on fingers
whose source angles violate the target box. Palm and both wrists remain exact.
All candidate adoption gates are evaluated against the declared projection.
"""
import time
import numpy as np
from scipy.optimize import least_squares, minimize

METHOD_ID = 'dga_practical_fixed_palm_branch_projection_pareto_uncapped_v3'
PARTS = ('palm', 'index', 'middle', 'ring', 'little', 'thumb', 'wrist_carrier')
PREFIX = {'index':'FF', 'middle':'MF', 'ring':'RF', 'little':'LF', 'thumb':'TH'}
SOLVER = {'position_max_nfev':80, 'geometry_maxiter':30,
          'geometry_eps_rad':1e-5, 'geometry_ftol':1e-10,
          'max_unique_field_states_per_branch':64,
          'finger_distance_cap_m':None, 'objective_normalization_m':.001}


def explicit_box_projection(values, bounds):
    q, b = np.asarray(values, dtype=np.float64), np.asarray(bounds, dtype=np.float64)
    if (q.ndim != 1 or b.shape != (len(q),2) or not np.isfinite(q).all()
            or not np.isfinite(b).all() or np.any(b[:,0] > b[:,1])):
        raise ValueError('Finite ordered box and vector required')
    result = np.minimum(np.maximum(q, b[:,0]), b[:,1])
    return result, np.flatnonzero(result != q)


def branch_metrics(state, source, *, site_indices, part_index):
    errors = np.asarray(state['errors_m'])[site_indices]
    delta = np.asarray(state['fields'])[:,part_index] - source[:,part_index]
    if errors.ndim != 2 or errors.shape[1] != 3 or not len(errors):
        raise ValueError('Complete corresponding branch landmarks required')
    if not np.isfinite(errors).all() or not np.isfinite(delta).all():
        raise ValueError('Finite branch metrics required')
    return {'maximum_landmark_error_m':float(np.linalg.norm(errors,axis=1).max()),
            'landmark_SSE_m2':float(np.sum(errors * errors)),
            'maximum_unsigned_field_error_m':float(np.max(np.abs(delta))),
            'unsigned_field_MSE_m2':float(np.mean(delta * delta))}


def allowed(metrics, baseline, *, strict=False):
    keys = ('maximum_landmark_error_m', 'landmark_SSE_m2', 'maximum_unsigned_field_error_m')
    finite = all(np.isfinite(v) and v >= 0 for v in metrics.values())
    return (finite and all(metrics[k] <= baseline[k] for k in keys)
            and (not strict or any(metrics[k] < baseline[k] for k in keys)))


def preference(metrics):
    # Aggregate fitting loss never overrides either maximum-error guard.
    return (metrics['maximum_unsigned_field_error_m'],
            metrics['maximum_landmark_error_m'],
            metrics['unsigned_field_MSE_m2'], metrics['landmark_SSE_m2'])


def domain_valid(state):
    return all(state.get(k) is True for k in ('strict_domain_valid',
        'original_wrists_preserved','rigid_frames_valid','independent_fk_valid'))


def summarize(state):
    keys = ('variables','q','palm','delta','root','forearm','errors_m',
            'landmark_metrics','target_fk_max_entry','palm_chain_max_entry',
            'strict_domain_valid','original_wrists_preserved',
            'rigid_frames_valid','independent_fk_valid')
    return {k:state[k] for k in keys}


def map_pose(ctx, source):
    """No global row ID enters this reusable mapping routine.

    The caller binds raw provenance, source geometry and query dependencies.
    Source-invalid inputs must be rejected before construction of ctx. A solver
    failure preserves the explicitly labelled projection baseline, with no
    claim of corrected native grasp equivalence.
    """
    before = ctx.source_identity()
    source = np.asarray(source)
    if source.shape != (len(ctx.queries),8) or not np.isfinite(source).all():
        raise ValueError('Complete frozen finite source fields required')
    bounds = np.asarray([[ctx.target_model['joints'][n]['limit'][k]
                          for k in ('lower','upper')] for n in ctx.fingers])
    original = np.array([ctx.qsource[n] for n in ctx.fingers])
    projected, changed = explicit_box_projection(original, bounds)
    x0 = np.r_[projected, np.zeros(6)]
    baseline = ctx.make_state(x0, None, 'palm')
    if not domain_valid(baseline): raise ValueError('Projection baseline domain failure')
    baseline['fields'] = ctx.evaluate_fields(baseline)
    current = baseline
    reference_branches = []
    trace = []
    selected_branch_fields = {}
    for part, prefix in PREFIX.items():
        indices = [i for i,n in enumerate(ctx.fingers) if n.startswith(prefix)]
        sites = [i for i,n in enumerate(ctx.sites) if n.startswith(prefix.lower())]
        part_index = PARTS.index(part)
        reference = branch_metrics(baseline, source, site_indices=sites, part_index=part_index)
        offending = any(i in changed for i in indices)
        reference_branches.append({'part':part,'source_target_domain_violation':offending,
                                   'reference_metrics':reference})
        if not offending:
            trace.append({'part':part,'status':'IDENTITY_BRANCH_RETAINED',
                          'baseline_metrics':reference,'selected_metrics':reference,
                          'adopted_correction':False,'field_states':0})
            continue
        start = current
        start_q = start['variables'][indices].copy()
        start_time = time.monotonic()
        cache = {}
        candidates = []
        attempts = []
        trial_log = []
        optimizer_exception = None
        limit = SOLVER['max_unique_field_states_per_branch']
        class Budget(Exception): pass

        def state_for(q):
            q = np.asarray(q,dtype=np.float64)
            if q.shape != (len(indices),) or not np.isfinite(q).all():
                raise ValueError('Malformed branch trial')
            if np.any(q < bounds[indices,0]) or np.any(q > bounds[indices,1]):
                raise ValueError('Target-domain trial refused; no final repair')
            x = start['variables'].copy(); x[indices] = q
            state = ctx.make_state(x,start,part)
            if not domain_valid(state): raise ValueError('Branch trial certification failed')
            return state

        def field_state(q):
            key = np.asarray(q,dtype=np.float64).tobytes()
            if key in cache: return cache[key]
            if len(trial_log) >= limit: raise Budget('BRANCH_FIELD_STATE_BUDGET')
            state = state_for(q)
            state['fields'] = ctx.evaluate_fields(state,start,part)
            metrics = branch_metrics(state,source,site_indices=sites,part_index=part_index)
            state['branch_metrics'] = metrics
            eligible = allowed(metrics,reference,strict=True)
            if eligible: candidates.append(state)
            trial_log.append({'scope':'OPTIMIZER_WORKSPACE','branch_q':np.asarray(q).tolist(),
                'metrics':metrics,'true_projection_reference_guards_pass':allowed(metrics,reference),
                'strict_improvement':eligible})
            cache[key] = state
            return state

        # All objective terms are displacement from the original source; none
        # attracts the hand toward an object or rewards a successful grasp.
        def positional_residual(q):
            return state_for(q)['errors_m'][sites].reshape(-1) / .001

        seeds = [('projection',start_q)]
        try:
            positional = least_squares(positional_residual,start_q,
                bounds=(bounds[indices,0],bounds[indices,1]),method='trf',jac='3-point',
                max_nfev=SOLVER['position_max_nfev'],ftol=1e-11,xtol=1e-11,gtol=1e-11)
            seeds.append(('position_fit',positional.x))
            field_state(positional.x)
            attempts.append({'stage':'position_seed','success':bool(positional.success),
                'message':str(positional.message),'reported_nfev':int(positional.nfev)})
        except Budget as error:
            attempts.append({'stage':'position_seed','status':str(error)})
        except (ValueError, FloatingPointError, RuntimeError) as error:
            optimizer_exception = {'type':type(error).__name__,'message':str(error)}
            attempts.append({'stage':'position_seed','status':'EXCEPTION_BASELINE_FALLBACK',
                             'exception':optimizer_exception})

        def objective(q):
            metrics = field_state(q)['branch_metrics']
            return .5 * (metrics['landmark_SSE_m2']/len(sites)
                          + metrics['unsigned_field_MSE_m2']) / .001**2

        def inequalities(q):
            metrics = field_state(q)['branch_metrics']
            return np.array([(reference['maximum_landmark_error_m']-metrics['maximum_landmark_error_m'])/.001,
                (reference['landmark_SSE_m2']-metrics['landmark_SSE_m2'])/(len(sites)*.001**2),
                (reference['maximum_unsigned_field_error_m']-metrics['maximum_unsigned_field_error_m'])/.001])

        for label, seed in seeds:
            if optimizer_exception is not None:
                attempts.append({'stage':'geometry_fit','start':label,'status':'SKIPPED_AFTER_EXCEPTION'})
                continue
            try:
                result = minimize(objective,seed,method='SLSQP',
                    bounds=[tuple(b) for b in bounds[indices]],
                    constraints=[{'type':'ineq','fun':inequalities}],
                    options={'maxiter':SOLVER['geometry_maxiter'],
                             'eps':SOLVER['geometry_eps_rad'],
                             'ftol':SOLVER['geometry_ftol'],'disp':False})
                field_state(result.x)
                attempts.append({'stage':'geometry_fit','start':label,'status':'COMPLETE',
                    'success':bool(result.success),'message':str(result.message),
                    'iterations':int(result.nit),'reported_nfev':int(result.nfev)})
            except Budget as error:
                attempts.append({'stage':'geometry_fit','start':label,'status':str(error)})
            except (ValueError, FloatingPointError, RuntimeError) as error:
                optimizer_exception = {'type':type(error).__name__,'message':str(error)}
                attempts.append({'stage':'geometry_fit','start':label,'status':'EXCEPTION_BASELINE_FALLBACK',
                                 'exception':optimizer_exception})
        # Ordinary fixed budget/iteration termination leaves mathematically
        # verified trial proposals available. An exception invalidates this
        # branch attempt and selects the original explicit reference instead.
        chosen = min(candidates,key=lambda s:preference(s['branch_metrics'])) if candidates and optimizer_exception is None else start
        selected_metrics = branch_metrics(chosen,source,site_indices=sites,part_index=part_index)
        if not allowed(selected_metrics,reference): raise ValueError('Final projection-reference guard failure')
        if any(chosen['q'][n] != start['q'][n] for n in ctx.names if not n.startswith(prefix)):
            raise ValueError('A correction changed another branch')
        current = chosen
        selected_branch_fields[part] = chosen['fields'][:,part_index].copy()
        trace.append({'part':part,'status':'BOUNDED_CORRECTION' if chosen is not start else 'PROJECTION_REFERENCE_FALLBACK',
            'baseline_metrics':reference,'selected_metrics':selected_metrics,
            'adopted_correction':chosen is not start,'field_states':len(trial_log),
            'elapsed_seconds':time.monotonic()-start_time,'attempts':attempts,'trials':trial_log,
            'optimizer_exception':optimizer_exception})

    # Full final recomputation independently challenges the partial-field cache.
    full = ctx.evaluate_fields(current)
    if not np.array_equal(full,current['fields']):
        raise ValueError('Partial/full target-field replay mismatch')
    current['fields'] = full
    for item in reference_branches:
        part=item['part']; prefix=PREFIX[part]
        metrics=branch_metrics(current,source,
            site_indices=[i for i,n in enumerate(ctx.sites) if n.startswith(prefix.lower())],
            part_index=PARTS.index(part))
        if not allowed(metrics,item['reference_metrics']): raise ValueError('Final branch regression')
        if not item['source_target_domain_violation']:
            if any(current['q'][n] != ctx.qsource[n] for n in ctx.names if n.startswith(prefix)):
                raise ValueError('Legal native branch changed')
    if not np.array_equal(current['palm'],ctx.source_palm): raise ValueError('Source palm changed')
    if not np.array_equal(current['variables'][22:],np.zeros(6)): raise ValueError('Nonidentity palm variables')
    if any(current['q'][n] != ctx.qsource[n] for n in ('WRJ2','WRJ1')): raise ValueError('Wrist changed')
    if ctx.source_identity() != before: raise ValueError('Native identity changed')
    ctx.verify_immutable_source(); ctx.certify_payload(current)
    return baseline,current,{'method_id':METHOD_ID,'projection_changed_joint_names':[ctx.fingers[i] for i in changed],
        'baseline_definition':'Explicit unique minimum Euclidean finger-joint change into target box, fixed native palm/wrists',
        'identity_pose':not len(changed),'branches':trace,
        'full_partial_field_replay_exact':True,'surface_field_scope':'856 frozen fitting queries; five finger unsigned visual fields uncapped; palm/wrist unsigned and signed palm diagnostics retain 4 mm cap',
        'fidelity_acceptance':'NOT_CLAIMED','execution_eligible':False}
