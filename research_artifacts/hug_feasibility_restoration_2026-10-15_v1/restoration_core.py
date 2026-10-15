"""FK-only restoration of frozen endpoints; exact gates cannot be relaxed."""
import hashlib
from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize

class BudgetStop(RuntimeError):pass
class InvalidState(RuntimeError):pass

def key(x):
    x=np.asarray(x,np.float64)
    if x.shape!=(27,) or not np.isfinite(x).all():raise InvalidState('finite complete27 state required')
    return x.tobytes()

def tip_status(errors,caps):
    errors=np.asarray(errors,float);caps=np.asarray(caps,float)
    if errors.shape!=(5,3) or caps.shape!=(5,) or not np.isfinite(errors).all() or not np.isfinite(caps).all() or np.any(caps<0):raise ValueError('five finite native tip vectors/caps required')
    norms=np.sqrt(np.einsum('ij,ij->i',errors,errors));mean=float(norms.mean())
    return {'norms_m':norms,'mean_m':mean,'individual_pass':norms<=caps,'mean_pass':mean<=.010,'passed':bool(np.all(norms<=caps) and mean<=.010)}

def norm_jacobian(errors,J):
    norms=np.sqrt(np.einsum('ij,ij->i',errors,errors));out=np.zeros((5,27));use=norms>0
    out[use]=np.einsum('ij,ijk->ik',errors[use],J[use])/norms[use,None]
    return out

def surface_gate(value,reference_rank,protected_valid,protected_pass):
    valid=np.asarray(value['exact_orientation_fields']['valid'],bool);passed=np.asarray(value['exact_orientation']['passed'],bool)
    pv=np.asarray(protected_valid,bool);pp=np.asarray(protected_pass,bool)
    if not valid.shape==passed.shape==pv.shape==pp.shape:raise ValueError('All fixed orientation rows must remain present')
    rank_values=np.asarray(value['rank'],float);reference_values=np.asarray(reference_rank,float)
    if rank_values.shape!=(5,) or reference_values.shape!=(5,) or not np.isfinite(rank_values).all() or not np.isfinite(reference_values).all():raise ValueError('Complete finite five-component exact ranks required')
    lost_valid=pv&~valid;lost_pass=pp&~passed;rank=tuple(value['rank'])
    reasons=[]
    if not value['in_domain']:reasons.append('original_domain_failed')
    if not value['tip_status']['passed']:reasons.append('exact_original_tip_or_mean_failed')
    if lost_valid.any():reasons.append('previously_valid_orientation_row_invalidated')
    if lost_pass.any():reasons.append('previously_passed_orientation_row_failed')
    if not rank<tuple(reference_rank):reasons.append('no_strict_original_rank_improvement')
    return {'accepted_for_limited_coarse_audit':not reasons,'refusal_reasons':reasons,'lost_orientation_valid_rows':np.flatnonzero(lost_valid),'lost_orientation_pass_rows':np.flatnonzero(lost_pass),'exact_rank':rank,'reference_rank':tuple(reference_rank),'scientific_mapping_acceptance':False}

def project_endpoint(proposal,original_bounds,caps,evaluate_tips,evaluate_jacobian,config,emit):
    proposal=np.asarray(proposal,float).copy();key(proposal);domain=np.asarray(original_bounds,float);caps=np.asarray(caps,float)
    if domain.shape!=(27,2) or not np.isfinite(domain).all() or np.any(domain[:,0]>=domain[:,1]) or np.any(proposal<domain[:,0]) or np.any(proposal>domain[:,1]):raise ValueError('Finite ordered original bounds containing unchanged endpoint required')
    radius=config['trust_half_width_scaled']
    if not np.isfinite(radius) or radius<=0:raise ValueError('Positive finite trust half-width required')
    bounds=np.column_stack((np.maximum(domain[:,0],proposal-radius),np.minimum(domain[:,1],proposal+radius)))
    assert radius>0 and np.all(bounds[:,0]<bounds[:,1])
    cache=OrderedDict();counts={'FK_executions':0,'analytic_jacobian_executions':0,'FK_cache_hits':0,'selection_checkpoint_count':0};eligible=[];events=0
    def event(kind,**v):
        nonlocal events
        events+=1;emit({'event':events,'kind':kind,**v});return events
    def checked(x,fresh=False):
        x=np.asarray(x,float);k=key(x)
        if np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]):
            event('state_refusal',state_scaled=x,reason='outside_frozen_original_domain_or_endpoint_trust_box');raise InvalidState('out of trust/domain')
        if not fresh and k in cache:
            counts['FK_cache_hits']+=1;event('FK_cache_hit',state_scaled=x,source_event=cache[k][1]);return cache[k][0]
        limit=config['maximum_FK_executions_per_endpoint']-(0 if fresh else 1)
        if counts['FK_executions']>=limit:raise BudgetStop('FK budget reached')
        counts['FK_executions']+=1;supplied=evaluate_tips(x.copy());errors=np.asarray(supplied['error_vectors_m'],float).copy()
        status=tip_status(errors,caps);v={'state_scaled':x.copy(),'error_vectors_m':errors,'tip_status':status,'in_domain':bool(supplied['in_domain'])}
        ev=event('FK_callback',fresh=fresh,state_sha256=hashlib.sha256(k).hexdigest(),value=v)
        if not v['in_domain']:raise InvalidState('physical/original correction-domain refusal')
        cache[k]=(v,ev)
        if len(cache)>config['cache_size']:cache.popitem(last=False)
        return v
    def checkpoint(x,reason):
        v=checked(x);counts['selection_checkpoint_count']+=1
        ev=event('projection_checkpoint',reason=reason,state_scaled=x,tip_status=v['tip_status'],in_domain=v['in_domain'])
        if v['tip_status']['passed'] and v['in_domain']:
            eligible.append({'source_event':ev,'reason':reason,'state_scaled':np.asarray(x,float).copy(),'distance_squared_scaled':float(np.dot(x-proposal,x-proposal))})
        else:event('checkpoint_refusal',source_event=ev,reason='literal_original_tip_caps_or_mean_failed')
    cap_target=np.maximum(0.,caps-config['inward_target_m']);mean_target=.010-config['inward_target_m']
    def objective(x):d=x-proposal;return .5*float(np.dot(d,d))
    def gradient(x):return x-proposal
    def constraints(x):v=checked(x);n=v['tip_status']['norms_m'];return np.r_[(cap_target-n)/.001,(mean_target-n.mean())/.001]
    def jacobian(x):
        v=checked(x)
        if counts['analytic_jacobian_executions']>=config['maximum_analytic_jacobians_per_endpoint']:raise BudgetStop('analytic Jacobian budget reached')
        counts['analytic_jacobian_executions']+=1;J=np.asarray(evaluate_jacobian(np.asarray(x,float).copy()),float)
        if J.shape!=(5,3,27) or not np.isfinite(J).all():raise ValueError('finite virtual-tip27D Jacobian required')
        NJ=norm_jacobian(v['error_vectors_m'],J);out=np.vstack((-NJ/.001,-NJ.mean(0)/.001))
        event('analytic_jacobian',state_scaled=x,matrix=out,zero_norm_rows=np.flatnonzero(v['tip_status']['norms_m']==0));return out
    major=0
    def callback(x):
        nonlocal major
        major+=1;checkpoint(x,f'major_{major}')
    optimizer=None;termination='completed';refusal=None
    try:
        checkpoint(proposal,'frozen_endpoint')
        result=minimize(objective,proposal.copy(),jac=gradient,method='SLSQP',bounds=[tuple(v) for v in bounds],constraints=[{'type':'ineq','fun':constraints,'jac':jacobian}],callback=callback,options={'maxiter':config['maximum_iterations_per_endpoint'],'ftol':config['optimizer_ftol'],'disp':False})
        optimizer={'status':int(result.status),'success':bool(result.success),'message':str(result.message),'nit':int(result.nit),'nfev':int(result.nfev),'njev':int(result.njev),'terminal_state_scaled':result.x.copy(),'objective':float(result.fun)}
        event('optimizer_terminal',**optimizer);checkpoint(result.x,'optimizer_terminal')
    except (BudgetStop,InvalidState) as exc:
        termination=type(exc).__name__;refusal=str(exc);event('projection_stopped',reason=refusal)
    selected=None;fresh=None
    if eligible:
        selected=min(eligible,key=lambda z:z['distance_squared_scaled'])
        try:
            fresh=checked(selected['state_scaled'],fresh=True);event('selected_fresh_FK',source_checkpoint=selected,value=fresh)
            if not fresh['in_domain'] or not fresh['tip_status']['passed']:
                event('fresh_selection_refusal',source_checkpoint=selected,value=fresh);selected=None
        except (BudgetStop,InvalidState) as exc:
            event('fresh_selection_refusal',source_checkpoint=selected,reason=str(exc));selected=None
    event('completed',counts=counts,eligible_checkpoints=len(eligible),termination=termination)
    return {'selected':selected,'fresh_FK':fresh,'optimizer':optimizer,'termination':termination,'refusal':refusal,'counts':counts,'eligible_checkpoints':eligible,'trust_bounds_scaled':bounds,'proposal_state_scaled':proposal,'final_event_count':events,'surface_geometry_calls_during_projection':0,'scientific_mapping_acceptance':False}
