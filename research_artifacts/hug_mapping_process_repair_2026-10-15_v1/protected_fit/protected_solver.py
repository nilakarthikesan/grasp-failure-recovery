"""Finite minimax search over static witnesses; exact checked selection only."""
from pathlib import Path
import gzip,hashlib,json,time
from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize
from protected_surface_policy import protected_surface_constraint_policy
from se3_case import base
from se3_math import decode_state,ROTATION_COMPONENT_LIMIT_RAD
plain=base.plain

class BudgetStop(RuntimeError):pass

def state_key(x):
    x=np.asarray(x,dtype=np.float64)
    if x.shape!=(30,) or not np.isfinite(x).all():raise ValueError('Complete finite30-state required')
    return x.tobytes()

def norm_jacobian(errors,jacobian):
    errors=np.asarray(errors,float);J=np.asarray(jacobian,float)
    norms=np.sqrt(np.einsum('ij,ij->i',errors,errors))
    out=np.zeros((5,30));nonzero=norms>0
    out[nonzero]=np.einsum('ij,ijk->ik',errors[nonzero],J[nonzero])/norms[nonzero,None]
    return out

def solve(case,starts,configuration,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tracepath=out/f'case_{case.index}_trace.jsonl.gz'
    if tracepath.exists():raise FileExistsError('Never overwrite study evidence')
    raw_limit=configuration['maximum_raw_callback_executions_per_case']
    exact_limit=configuration['maximum_exact_checks_per_case']
    reserve=configuration['terminal_raw_and_exact_reserve']
    maxiter=configuration['maximum_iterations_per_start'];fdstep=configuration['surface_difference_step_scaled']
    assert len(starts)==1 and raw_limit==1000 and exact_limit==30 and reserve==2 and maxiter==25
    cache=OrderedDict();cache_hits=0;eventseq=0;eligible=[];exact_records=[];start_records=[];selected=None;reference=None
    counters0=(case.raw_calls,case.exact_calls,case.tip_jacobian_calls)
    begun=time.perf_counter()
    stream=gzip.open(tracepath,'wt',encoding='utf-8')
    def emit(kind,**data):
        nonlocal eventseq
        eventseq+=1
        stream.write(json.dumps(plain({'event':eventseq,'kind':kind,**data}),separators=(',',':'),allow_nan=False)+'\n')
        stream.flush()
        return eventseq
    def limits(reserved=True):
        if reserved and time.perf_counter()-begun>=configuration['wall_clock_limit_s']:raise BudgetStop('wall clock budget reached')
        if case.raw_calls-counters0[0]>=raw_limit-(reserve if reserved else 0):raise BudgetStop('raw callback budget reached')
    def raw(x):
        nonlocal cache_hits
        key=state_key(x)
        if key in cache:
            cache_hits+=1;emit('raw_cache_hit',state_scaled=x,state_sha256=hashlib.sha256(key).hexdigest(),source_event=cache[key][1])
            return cache[key][0]
        limits();value=case.raw(x)
        ev=emit('raw_callback',state_sha256=hashlib.sha256(key).hexdigest(),value=value)
        cache[key]=(value,ev)
        # Bounded storage, without deleting any losslessly saved trace evidence.
        if len(cache)>configuration['raw_cache_size']:cache.popitem(last=False)
        return value
    def check(x,reason,reserved=True,already_raw=None):
        nonlocal selected
        x=np.asarray(x,float);key=state_key(x)
        r=raw(x) if already_raw is None else already_raw
        q,shift,omega=decode_state(x,case.q0)
        aabb=np.asarray(case.domain['world_displacement_AABB_m'])
        decoded_domain=bool(np.all(shift>=aabb[:,0]) and np.all(shift<=aabb[:,1]) and np.all(np.abs(omega)<=ROTATION_COMPONENT_LIMIT_RAD))
        domain=decoded_domain and bool(np.all(x>=case.bounds[:,0]) and np.all(x<=case.bounds[:,1]) and np.all(q>=case.limits[:,0]) and np.all(q<=case.limits[:,1]) and np.all(q>=case.q0-.35) and np.all(q<=case.q0+.35))
        if not domain or not r['tip_status']['passed']:
            emit('selection_refusal_before_orientation',reason=reason,state_scaled=x,in_domain=domain,tip_status=r['tip_status'],orientation_not_computed=True)
            return None
        limits(reserved)
        if case.exact_calls-counters0[1]>=exact_limit-(reserve if reserved else 0):raise BudgetStop('exact checkpoint budget reached')
        full=case.exact(x)
        assert full['in_domain'] and full['tip_status']['passed']
        ev=emit('exact_checkpoint',reason=reason,value=full)
        entry={'event':ev,'reason':reason,'state_scaled':x.copy(),'rank':tuple(full['rank']),
               'all_witness_constraints_pass':full['all_witness_constraints_pass'],
               'invalid_orientation_rows':int((~full['exact_orientation_fields']['valid']).sum()),
               'failed_proximity_rows':int((~full['proximity']['passed']).sum()),
               'failed_orientation_rows':int((~full['exact_orientation']['passed']).sum())}
        exact_records.append(entry)
        # Only source-labelled starts, solver major iterates or terminals are
        # eligible. Raw finite-difference/line-search states cannot enter here.
        protected = reference is None or (np.all(full['exact_orientation_fields']['valid'][reference['exact_orientation_fields']['valid']]) and np.all(full['literal_surface_pass'][reference['literal_surface_pass']]))
        entry['preserves_reference_eligible_and_passing_rows']=bool(protected)
        if protected:
            eligible.append((entry,full))
            if selected is None or tuple(entry['rank'])<tuple(selected[0]['rank']):selected=(entry,full)
        else:emit('protected_reference_refusal',source_event=ev,reason='lost previously eligible orientation or passed retained row')
        return full

    reference=check(np.asarray(starts[0]['state_scaled'],float),'fixed_pooled_selection_reference')
    assert reference is not None
    baseline=check(np.zeros(30),'original_baseline')
    assert baseline is not None
    policy=protected_surface_constraint_policy(reference)
    offsets=np.asarray(policy['offset'],float);slack=np.asarray(policy['slack_coefficients'],float)
    assert offsets.shape==slack.shape==np.asarray(reference['surface_margin']).shape
    assert np.all((slack==0.) | (slack==1.))
    assert np.all(np.asarray(reference['surface_margin'])[slack==0.]+offsets[slack==0.]>=0.), 'Frozen reference must satisfy every hard raw constraint'
    emit('protected_surface_policy',policy=policy)
    termination='one_frozen_pooled_start_completed'
    for start_index,start in enumerate(starts):
        x0=np.asarray(start['state_scaled'],float)
        major_count=0
        try:
            start_raw=raw(x0);check(x0,f'start_{start_index}',already_raw=start_raw)
            t0=max(0.,float(np.max(-(np.asarray(start_raw['surface_margin'])+offsets)[slack==1.],initial=0.)))+1e-6
            assert t0<configuration['epigraph_slack_upper']
            z0=np.r_[x0,t0]
            emit('optimizer_start',start_index=start_index,label=start['label'],state_scaled=x0,initial_epigraph=t0)
            jac_cache={}
            def fun(z):return float(z[-1])
            def jac(z):v=np.zeros(31);v[-1]=1.;return v
            def constraints(z):
                v=raw(z[:30]);norms=v['tip_status']['norms_m']
                cap=np.maximum(0.,case.caps-configuration['tip_inward_target_m'])
                return np.r_[(cap-norms)/.001,(.010-configuration['tip_inward_target_m']-norms.mean())/.001,v['surface_margin']+offsets+slack*z[-1]]
            def cjac(z):
                x=np.asarray(z[:30]);key=state_key(x)
                if key in jac_cache:return jac_cache[key]
                base=raw(x);J=case.tip_jacobian(x);nj=norm_jacobian(base['task_error_vectors_m'][:5],J)
                sm=base['surface_margin'];surface=np.empty((len(sm),30));spans=[]
                for k in range(30):
                    probe=x.copy();forward=min(case.bounds[k,1],x[k]+fdstep)-x[k]
                    backward=x[k]-max(case.bounds[k,0],x[k]-fdstep)
                    delta=forward if forward>=backward else -backward
                    if delta==0:raise ValueError('Zero finite-difference span')
                    probe[k]+=delta;surface[:,k]=(raw(probe)['surface_margin']-sm)/delta;spans.append(delta)
                result=np.zeros((6+len(sm),31));result[:5,:30]=-nj/.001;result[5,:30]=-nj.mean(0)/.001
                result[6:,:30]=surface;result[6:,-1]=slack
                ev=emit('constraint_jacobian',state_scaled=x,one_sided_spans_scaled=spans,matrix=result,zero_tip_norm_rows=np.flatnonzero(base['tip_status']['norms_m']==0))
                jac_cache.clear();jac_cache[key]=result
                return result
            def callback(z):
                nonlocal major_count
                major_count+=1
                emit('optimizer_major_iterate',start_index=start_index,major_iteration=major_count,state_scaled=z[:30],epigraph=float(z[-1]))
                check(z[:30],f'start_{start_index}_major_{major_count}')
            result=minimize(fun,z0,jac=jac,method='SLSQP',bounds=[tuple(v) for v in case.bounds]+[(0.,configuration['epigraph_slack_upper'])],
                constraints=[{'type':'ineq','fun':constraints,'jac':cjac}],callback=callback,
                options={'maxiter':maxiter,'ftol':configuration['optimizer_ftol'],'disp':False})
            saved={'start_index':start_index,'status':int(result.status),'success':bool(result.success),'message':str(result.message),'nit':int(result.nit),'nfev':int(result.nfev),'njev':int(result.njev),'terminal_state_scaled':result.x[:30].copy(),'terminal_epigraph':float(result.x[-1]),'major_iterations_observed':major_count}
            emit('optimizer_terminal',**saved);start_records.append(saved)
            check(result.x[:30],f'start_{start_index}_terminal')
        except BudgetStop as exc:
            termination=str(exc);emit('budget_stop',start_index=start_index,reason=str(exc),major_iterations_observed=major_count)
            start_records.append({'start_index':start_index,'budget_stop':str(exc),'major_iterations_observed':major_count})
            break
    assert selected is not None
    chosen_entry,chosen=selected
    # A fresh terminal check is outside every optimizer/cache. It consumes one
    # of the two reserved raw/exact executions and cannot promote a raw probe.
    chosen_x=chosen_entry['state_scaled'].copy();limits(False)
    if case.exact_calls-counters0[1]>=exact_limit:raise BudgetStop('terminal exact reserve violated')
    fresh=case.exact(chosen_x);terminal_event=emit('selected_fresh_terminal',selected_source_event=chosen_entry['event'],value=fresh)
    refused_terminal_source=None
    fresh_protected=bool(np.all(fresh['exact_orientation_fields']['valid'][reference['exact_orientation_fields']['valid']]) and np.all(fresh['literal_surface_pass'][reference['literal_surface_pass']]))
    same_predicates=bool(np.array_equal(fresh['exact_orientation_fields']['valid'],chosen['exact_orientation_fields']['valid']) and np.array_equal(fresh['literal_surface_pass'],chosen['literal_surface_pass']))
    if not fresh['in_domain'] or not fresh['tip_status']['passed'] or tuple(fresh['rank'])!=tuple(chosen_entry['rank']) or not fresh_protected or not same_predicates:
        emit('fresh_terminal_refusal',reason='fresh tip/domain/rank differs from selected exact evidence',selected_state_scaled=chosen_x)
        chosen_x=np.asarray(starts[0]['state_scaled'],float).copy();limits(False)
        if case.exact_calls-counters0[1]>=exact_limit:raise BudgetStop('fallback exact reserve violated')
        refused_terminal_source=chosen_entry
        fresh=case.exact(chosen_x);terminal_event=emit('fresh_reference_fallback',value=fresh)
        assert fresh['in_domain'] and fresh['tip_status']['passed']
        assert np.array_equal(fresh['exact_orientation_fields']['valid'],reference['exact_orientation_fields']['valid'])
        assert np.array_equal(fresh['literal_surface_pass'],reference['literal_surface_pass'])
        chosen_entry={'event':terminal_event,'reason':'fresh_reference_fallback','state_scaled':chosen_x.copy(),
            'rank':tuple(fresh['rank']),'all_witness_constraints_pass':fresh['all_witness_constraints_pass'],
            'invalid_orientation_rows':int((~fresh['exact_orientation_fields']['valid']).sum()),
            'failed_proximity_rows':int((~fresh['proximity']['passed']).sum()),
            'failed_orientation_rows':int((~fresh['exact_orientation']['passed']).sum())}
    stream.close()
    counts={'raw_callback_executions':case.raw_calls-counters0[0],'exact_checkpoint_executions':case.exact_calls-counters0[1],
            'analytic_tip_jacobian_executions':case.tip_jacobian_calls-counters0[2],
            'raw_cache_hits':cache_hits,'trace_events':eventseq,'selection_eligible_exact_checks':len(eligible),
            'raw_limit_includes_exact_callback_raw_work':True,'cache_hits_count_towards_limit':False,'reserved_terminal_executions':reserve}
    assert counts['raw_callback_executions']<=raw_limit and counts['exact_checkpoint_executions']<=exact_limit
    return chosen_x,fresh,{'schema':'hug_se3_hard_protected_minimax_solver_audit_v1','protected_surface_policy':policy,'configuration':configuration,
        'frozen_starts':starts,'baseline_exact':baseline,'protected_pooled_reference_exact':reference,'selected_source_exact_checkpoint':chosen_entry,
        'exact_checkpoint_summaries':exact_records,'optimizer_starts':start_records,'termination':termination,
        'refused_preterminal_selection':refused_terminal_source,'actual_returned_fresh_event':terminal_event,
        'counts':counts,'trace_path':str(tracepath),'trace_sha256':hashlib.sha256(tracepath.read_bytes()).hexdigest(),
        'runtime_seconds':time.perf_counter()-begun,'solver_convergence_not_scientific_acceptance':True,
        'all_witnesses_not_full_grid_coverage':True,'no_new_eval_data':True,'no_physics':True}
