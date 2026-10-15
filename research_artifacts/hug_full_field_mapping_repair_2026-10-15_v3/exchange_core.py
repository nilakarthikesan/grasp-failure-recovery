"""Bounded active-constraint exchange with full-field, exact admission.

Scene methods are injected. A local minimax proposal is never a returned pose
until every full-field baseline-passing relationship and original orientation
witness eligibility/sign protection passes. This is not global infeasibility.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

MAX_ROUNDS = 3
MAX_FULL_CHECKPOINTS = 6
MAX_LOCAL_MODEL_ROWS = 512 # six tip/mean rows plus all selected surface rows
DYADIC_EXPONENTS = tuple(range(11))
RESTORE_MAX_ITERATIONS = 8
RESTORE_MAX_CALLBACKS = 400
RESTORE_ACTIVATION_MARGIN_M = 1e-7
RESTORE_INWARD_M = 1e-9
DIFFERENCE_STEP_SCALED = 1e-4
TRUST_SCALED = .25
SAME_PART_ADMISSION = dict(schema_version='hug_full_field_same_part_admission_v3',
    minimum_newly_preserved_relationships=1,baseline_preserved_retention='all',
    current_preserved_retention='all',new_definite_violations_allowed=0,accepted_mapping=False,
    domain='complete_fixed_cell_by_same_anatomical_part_by_band_inventory',
    full_orientation_continuous_coverage='NOT_PROVED')


class RestorationBudgetStop(RuntimeError):
    pass


def tips_pass(state):
    return bool(np.all(state.literal_tip_caps_pass) and state.mean_tip_guard_pass
                and state.physical_joint_limits_pass and state.engineering_domain_pass)


def orientation_protected(baseline, candidate):
    for key in ('valid','passed'):
        a=np.asarray(baseline[key]);b=np.asarray(candidate[key])
        if a.dtype!=bool or b.dtype!=bool or a.shape!=b.shape:
            raise ValueError('Fixed exact orientation witness shape/type changed')
    return bool(np.all(~baseline['valid']|candidate['valid']) and
                np.all(~baseline['passed']|candidate['passed']))


def full_protection(baseline, candidate):
    a=baseline['fields'];b=candidate['fields']
    if a['preserved'].shape!=b['preserved'].shape:
        raise ValueError('Full original query domain changed')
    lost=a['preserved']&~b['preserved']
    original_definite=a['definite_lost']|a['definite_added']
    new=(b['definite_lost']|b['definite_added'])&~original_definite
    ori=orientation_protected(baseline['orientation'],candidate['orientation'])
    return dict(passed=bool(not lost.any() and not new.any() and ori),
                lost_baseline_passing=int(lost.sum()),new_definite_violations=int(new.sum()),
                orientation_protected=ori)


def full_rank(full):
    f=full['fields'];required=f['required'];margin=np.asarray(full['margin_m'],float)
    if margin.shape!=required.shape or not np.isfinite(margin).all():
        raise ValueError('Finite aligned full-field margins required')
    return (int((required&~f['preserved']).sum()),
            int((f['definite_lost']|f['definite_added']).sum()),
            float(max(0.,-margin[required].min(initial=0.))/.001))


def activate_worst(full,previous=()):
    """One worst unpreserved native-definite row/part/band/kind; stable ties."""
    fields=full['fields'];margin=np.asarray(full['margin_m'],float)
    if margin.shape!=fields['required'].shape or margin.ndim!=3 or margin.shape[1:]!=(6,3):
        raise ValueError('Complete six-part/three-band field required')
    active=set(tuple(map(int,row)) for row in previous)
    for part in range(6):
        for band in range(3):
            for near in (True,False):
                mask=fields['native_near' if near else 'native_far'][:,part,band]&~fields['preserved'][:,part,band]
                ids=np.flatnonzero(mask)
                if not len(ids):continue
                values=margin[ids,part,band]
                # argmin's first tied index is the lowest immutable cell ID.
                active.add((int(ids[int(np.argmin(values))]),part,band))
    return tuple(sorted(active))


def counterexample_rows(baseline,current,candidate):
    """Every literal full-field guard failure, with immutable tuple identity."""
    a=baseline['fields'];b=current['fields'];c=candidate['fields']
    definite=c['definite_lost']|c['definite_added']
    lost=(a['preserved']|b['preserved'])&~c['preserved']
    new=definite&(~(a['definite_lost']|a['definite_added'])|~(b['definite_lost']|b['definite_added']))
    return tuple(tuple(map(int,row)) for row in np.argwhere(lost|new))


def restoration_rows(baseline):
    """Untruncated 100nm SEARCH activation, never an acceptance allowance."""
    margin=np.asarray(baseline['margin_m'],float);preserved=baseline['fields']['preserved']
    if margin.shape!=preserved.shape or not np.isfinite(margin).all():
        raise ValueError('Complete finite baseline field required for restoration activation')
    return tuple(tuple(map(int,row)) for row in np.argwhere(preserved&(margin<=RESTORE_ACTIVATION_MARGIN_M)))


def active_policy(baseline,active,guard_rows=(),cell_error_m=None):
    cell,part,band=np.asarray(active,int).T
    preserved=baseline['fields']['preserved'][cell,part,band]
    definite=(baseline['fields']['definite_lost']|baseline['fields']['definite_added'])[cell,part,band]
    guards=set(guard_rows);guard=np.asarray([tuple(row) in guards for row in active],bool)
    far=baseline['fields']['native_far'][cell,part,band]
    unresolved_guard=guard&~preserved&~definite
    if unresolved_guard.any():
        error=np.asarray(cell_error_m,float)
        if error.shape!=(len(active),) or not np.isfinite(error).all() or np.any(error<1e-8):
            raise ValueError('Fixed radius plus numerical allowance required for unresolved guards')
    else:error=np.zeros(len(active))
    hard=preserved|unresolved_guard
    # A1nm inward search target on strict far rows never relaxes a threshold.
    offset=np.where(unresolved_guard,2*error/.001,0.)+np.where(far,-RESTORE_INWARD_M/.001,0.)
    return dict(hard_mask=hard,slack_coefficients=(~hard).astype(float),offset=offset,
                retained_counterexample_mask=guard,no_new_definite_guard_mask=unresolved_guard,
                full_preservation_hard_mask=preserved)


def same_part_receipt(baseline,returned):
    gate=full_protection(baseline,returned);a=baseline['fields'];b=returned['fields']
    repaired=int((b['preserved']&~a['preserved']).sum());tips=tips_pass(returned['state'])
    return dict(policy=dict(SAME_PART_ADMISSION),required_relationships=int(a['required'].sum()),
        baseline_preserved_relationships=int(a['preserved'].sum()),returned_preserved_relationships=int(b['preserved'].sum()),
        newly_preserved_relationships=repaired,all_baseline_preserved=gate['lost_baseline_passing']==0,
        lost_baseline_preserved_relationships=gate['lost_baseline_passing'],new_definite_violations=gate['new_definite_violations'],
        original_tip_caps_mean_and_domain_pass=tips,retained_orientation_protection_pass=gate['orientation_protected'],
        relationship_repair_credit=bool(repaired>=1 and gate['passed'] and tips),accepted_mapping=False)


def active_hard_pass(raw,policy):
    values=np.asarray(raw['surface_margin'],float)+policy['offset']
    if values.shape!=policy['hard_mask'].shape or not np.isfinite(values).all():
        raise ValueError('Aligned finite active raw margins required')
    return bool(np.all(values[policy['hard_mask']]>=0))


def restore_literal_caps(case,baseline,emit):
    """Joint cap and near-bound protection restoration in the original box."""
    selected=restoration_rows(baseline);calls=0
    def call(method,*args):
        nonlocal calls
        if calls>=RESTORE_MAX_CALLBACKS:raise RestorationBudgetStop('400 restoration callback limit reached')
        calls+=1;return method(*args)
    x0=np.zeros(30);initial=call(case.cheap,x0)
    emit('restoration_activation',complete_selected_rows=selected,selected_count=len(selected),
         search_activation_margin_m=RESTORE_ACTIVATION_MARGIN_M,acceptance_allowance=False,
         maximum_callbacks=RESTORE_MAX_CALLBACKS,maximum_iterations=RESTORE_MAX_ITERATIONS)
    if tips_pass(initial):
        emit('restoration_not_needed',state_scaled=x0)
        return x0,initial
    if len(selected)+6>MAX_LOCAL_MODEL_ROWS:
        emit('restoration_capacity_refusal',complete_selected_rows=selected,selected_count=len(selected),
             local_constraint_rows=len(selected)+6,maximum_local_model_rows=MAX_LOCAL_MODEL_ROWS,
             truncated=False,accepted_mapping=False)
        return None,initial
    bounds=case.bounds.copy()
    bounds[:,0]=np.maximum(bounds[:,0],-.01)
    bounds[:,1]=np.minimum(bounds[:,1],.01)
    target=np.maximum(0.,np.asarray(case.caps)-RESTORE_INWARD_M)
    cache={};jac_cache={}
    if selected:
        cell,part,band=np.asarray(selected,int).T
        far=baseline['fields']['native_far'][cell,part,band]
        surface_offset=np.where(far,-RESTORE_INWARD_M/.001,0.)
    else:surface_offset=np.empty(0)
    def at(x):
        key=np.asarray(x,np.float64).tobytes()
        if key not in cache:
            if selected:
                cache[key]=call(case.raw,x,selected)
            else:cache[key]=dict(state=call(case.cheap,x),surface_margin=np.empty(0))
        return cache[key]
    def constraint(x):
        raw=at(x);norm=raw['state'].tip_error_norms_m
        return np.r_[(target-norm)/.001,(.010-RESTORE_INWARD_M-norm.mean())/.001,
                     raw['surface_margin']+surface_offset]
    def jacobian(x):
        key=np.asarray(x,np.float64).tobytes()
        if key in jac_cache:return jac_cache[key]
        raw=at(x);state=raw['state'];J=call(case.tip_jacobian,x);errors=state.tip_errors_world_m
        norms=state.tip_error_norms_m;nj=np.zeros((5,30));valid=norms>0
        nj[valid]=np.einsum('ij,ijk->ik',errors[valid],J[valid])/norms[valid,None]
        sj=np.empty((len(selected),30))
        if selected:
            for k in range(30):
                forward=min(bounds[k,1],x[k]+DIFFERENCE_STEP_SCALED)-x[k]
                backward=x[k]-max(bounds[k,0],x[k]-DIFFERENCE_STEP_SCALED)
                span=forward if forward>=backward else -backward
                if span==0:raise ValueError('No restoration finite-difference span in declared narrow box')
                trial=x.copy();trial[k]+=span
                sj[:,k]=(at(trial)['surface_margin']-raw['surface_margin'])/span
        value=np.r_[-nj/.001,-nj.mean(0,keepdims=True)/.001,sj];jac_cache[key]=value
        return value
    try:
        result=minimize(lambda x:float(.5*np.dot(x,x)),x0,jac=lambda x:x,
            bounds=[tuple(row) for row in bounds],constraints=[dict(type='ineq',fun=constraint,jac=jacobian)],
            method='SLSQP',options=dict(maxiter=RESTORE_MAX_ITERATIONS,ftol=1e-12,disp=False))
    except RestorationBudgetStop as error:
        emit('restoration_budget_refusal',reason=str(error),restoration_callbacks=calls,
             selected_rows=selected,accepted_mapping=False)
        return None,initial
    x=np.asarray(result.x,float)
    if x.shape!=(30,) or not np.isfinite(x).all():raise ValueError('Malformed restoration return')
    if np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]):
        raise ValueError('Restoration return left its literal declared search box')
    try:
        if selected:
            final_raw=call(case.raw,x,selected);final=final_raw['state']
            final_margin=np.asarray(final_raw['surface_margin'],float)
            if final_margin.shape!=(len(selected),) or not np.isfinite(final_margin).all():
                raise ValueError('Finite aligned final joint-restoration margins required')
            selected_pass=np.where(far,final_margin>0,final_margin>=0)
            search_target_pass=final_margin+surface_offset>=0
        else:
            final=call(case.cheap,x);selected_pass=np.ones(0,bool);search_target_pass=np.ones(0,bool)
    except RestorationBudgetStop as error:
        emit('restoration_budget_refusal',reason=str(error),restoration_callbacks=calls,
             selected_rows=selected,accepted_mapping=False)
        return None,initial
    emit('restoration_result',state_scaled=x,optimizer_status=int(result.status),optimizer_success=bool(result.success),
         optimizer_message=str(result.message),iterations=int(result.nit),cheap_unique_evaluations=len(cache),
         restoration_callbacks=calls,selected_rows=selected,selected_count=len(selected),
         selected_original_relationship_pass=selected_pass,all_selected_original_relationships_pass=bool(selected_pass.all()),
         selected_inward_search_target_pass=search_target_pass,
         literal_caps_pass=tips_pass(final),inward_target_m=RESTORE_INWARD_M,
         acceptance_is_actual_literal_caps_not_optimizer_success=True)
    return (x,final) if tips_pass(final) and selected_pass.all() else (None,final)


def solve(case,local_minimax_step,configuration,emit):
    """At most baseline,start,three candidate,final full-field checkpoints."""
    baseline=case.original_full();baseline_rank=full_rank(baseline)
    x,state=restore_literal_caps(case,baseline,emit)
    if x is None:
        emit('repair_refused',reason='joint_cap_or_selected_surface_restoration_failed',baseline_rank=baseline_rank)
        return dict(status='REFUSED_NO_FEASIBLE_START',accepted_relationship_repair=False,accepted_mapping=False,
                    repaired_relationship_count=0,feasible_search_progress_only=False,baseline=baseline,returned=baseline,
                    rounds=[],activated_rows=[],retained_guard_counterexamples=[],
                    same_part_admission_receipt=same_part_receipt(baseline,baseline),
                    reason='joint_cap_or_selected_surface_restoration_failed')
    current=case.full(x,'restored_start')
    protection=full_protection(baseline,current)
    if not tips_pass(current['state']) or not protection['passed']:
        emit('repair_refused',reason='restored_start_loses_full_field_or_orientation_protection',protection=protection)
        return dict(status='REFUSED_NO_PROTECTED_START',accepted_relationship_repair=False,accepted_mapping=False,
                    repaired_relationship_count=0,feasible_search_progress_only=False,baseline=baseline,returned=baseline,
                    rounds=[],activated_rows=[],retained_guard_counterexamples=counterexample_rows(baseline,baseline,current),
                    same_part_admission_receipt=same_part_receipt(baseline,baseline),
                    reason='restored_start_loses_full_field_or_orientation_protection')
    active=activate_worst(current);rounds=[];accepted_states=[];last_probe=current;guards=set()
    for number in range(MAX_ROUNDS):
        active=activate_worst(last_probe,tuple(sorted(set(active)|guards)))
        emit('activation',round=number,complete_activated_rows=active,retained_guard_counterexamples=tuple(sorted(guards)),
             selection='all_retained_guard_counterexamples_then_worst_all_field_native_definite_per_part_band_kind')
        if not active:break
        if len(active)+6>MAX_LOCAL_MODEL_ROWS:
            row=dict(round=number,activated_rows=active,retained_guard_counterexamples=tuple(sorted(guards)),
                candidate_found=False,selected_feasible_search_step=False,accepted_relationship_repair=False,
                accepted_mapping=False,capacity_refusal=True,local_constraint_rows=len(active)+6,
                maximum_local_model_rows=MAX_LOCAL_MODEL_ROWS,truncated=False)
            rounds.append(row);emit('local_model_capacity_refusal',**row);break
        policy=active_policy(current,active,tuple(sorted(guards)),case.relationship_error_m(active))
        raw=case.raw(x,active);tip_J=case.tip_jacobian(x)
        J=np.empty((len(active),30));spans=[]
        for k in range(30):
            forward=min(case.bounds[k,1],x[k]+DIFFERENCE_STEP_SCALED)-x[k]
            backward=x[k]-max(case.bounds[k,0],x[k]-DIFFERENCE_STEP_SCALED)
            span=forward if forward>=backward else -backward
            if span==0:raise ValueError('No finite-difference span in fixed domain')
            trial=x.copy();trial[k]+=span
            J[:,k]=(case.raw(trial,active)['surface_margin']-raw['surface_margin'])/span
            spans.append(span)
        proposal=local_minimax_step(x,raw,J,tip_J,case.bounds,TRUST_SCALED/(2**number),case.caps,policy,configuration)
        row=dict(round=number,activated_rows=active,local_model=proposal,difference_spans_scaled=spans,
                 candidate_found=False,selected_feasible_search_step=False,accepted_relationship_repair=False,accepted_mapping=False)
        rounds.append(row);emit('local_model',**row)
        chosen=None
        if proposal['usable_finite_trust_step'] and np.any(proposal['step_scaled']!=0):
            for exponent in DYADIC_EXPONENTS:
                trial=x+(2.**(-exponent))*proposal['step_scaled']
                if np.any(trial<case.bounds[:,0]) or np.any(trial>case.bounds[:,1]):
                    emit('candidate_refused',round=number,exponent=exponent,reason='literal_domain');continue
                actual=case.raw(trial,active)
                if not tips_pass(actual['state']) or not active_hard_pass(actual,policy):
                    emit('candidate_refused',round=number,exponent=exponent,reason='literal_caps_or_active_hard_rows');continue
                orientation=case.orientation(trial)
                if not orientation_protected(baseline['orientation'],orientation):
                    emit('candidate_refused',round=number,exponent=exponent,reason='exact_orientation_witness_protection');continue
                chosen=trial;row.update(candidate_found=True,dyadic_exponent=exponent);break
        if chosen is None:continue
        full=case.full(chosen,f'round_{number}_candidate');last_probe=full
        gate=full_protection(baseline,full);current_gate=full_protection(current,full);rank=full_rank(full)
        repaired=int((full['fields']['preserved']&~current['fields']['preserved']).sum())
        allowed=tips_pass(full['state']) and gate['passed'] and current_gate['passed'] and rank<full_rank(current)
        found=counterexample_rows(baseline,current,full)
        if not allowed:guards.update(found)
        row.update(full_field_protection=gate,current_preserved_protection=current_gate,candidate_rank=rank,
            selected_feasible_search_step=bool(allowed),newly_preserved_relationships=repaired if allowed else 0,
            checkpoint_guard_counterexamples=found,retained_guard_counterexamples=tuple(sorted(guards)),
            accepted_relationship_repair=bool(allowed and repaired>0),accepted_mapping=False)
        emit('full_field_candidate',round=number,protection=gate,current_protection=current_gate,rank=rank,
            selected_feasible_search_step=bool(allowed),newly_preserved_relationships=repaired if allowed else 0,
            checkpoint_guard_counterexamples=found,retained_guard_counterexamples=tuple(sorted(guards)),
            accepted_relationship_repair=bool(allowed and repaired>0),accepted_mapping=False)
        if allowed:
            x=chosen.copy();current=full;accepted_states.append(x.copy())
    final=case.full(x,'fresh_final')
    gate=full_protection(baseline,final)
    masks_equal=all(np.array_equal(final['fields'][key],current['fields'][key]) for key in current['fields'])
    orientation_equal=all(np.array_equal(final['orientation'][key],current['orientation'][key]) for key in ('valid','passed','score','distance_m'))
    state_equal=all(np.array_equal(getattr(final['state'],key),getattr(current['state'],key))
                    for key in ('joint_qpos_rad','root_position_m','root_quaternion_wxyz','tip_error_norms_m'))
    distance_equal=np.array_equal(final['distance_m'],current['distance_m'])
    if (not tips_pass(final['state']) or not gate['passed'] or full_rank(final)!=full_rank(current)
            or not masks_equal or not orientation_equal or not state_equal or not distance_equal):
        raise ValueError('Final current-scene full-field/cap/orientation readback failed; no candidate returned')
    repaired=int((final['fields']['preserved']&~baseline['fields']['preserved']).sum())
    progressed=full_rank(final)<baseline_rank
    status='IMPROVED_RELATIONSHIPS_DIAGNOSTIC' if repaired else 'FEASIBLE_SEARCH_PROGRESS_ONLY' if progressed else 'NO_ACCEPTED_SURFACE_IMPROVEMENT'
    return dict(status=status,accepted_relationship_repair=bool(repaired>0),accepted_mapping=False,
                repaired_relationship_count=repaired,feasible_search_progress_only=bool(progressed and not repaired),
                baseline=baseline,returned=final,returned_state_scaled=x,rounds=rounds,activated_rows=active,
                retained_guard_counterexamples=tuple(sorted(guards)),same_part_admission_receipt=same_part_receipt(baseline,final),
                accepted_states=accepted_states,final_protection=gate,full_strict_mapping_accepted=False,
                full_orientation_coverage='NOT_PROVED',not_global_infeasibility=True)
