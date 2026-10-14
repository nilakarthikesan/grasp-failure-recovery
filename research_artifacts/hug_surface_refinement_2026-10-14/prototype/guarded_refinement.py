"""Deterministic engineering guard for a diagnostic surface candidate.

This is not a mapping acceptance rule. It retains the source joint target if
none of the frozen interpolants respects the EXISTING fingertip guard and
improves the frozen full-pose geometric objective. No optimizer is run here.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GuardConfig:
    alphas: tuple[float,...] = (1.,.5,.25,.125,.0625,.03125,0.)
    maximum_mean_fingertip_error_m: float = .010
    minimum_objective_improvement: float = 1e-12
    joint_delta_limit_rad: float = .35

    def __post_init__(self):
        a=np.asarray(self.alphas,float)
        if a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or a[0]!=1 or a[-1]!=0 or np.any(np.diff(a)>=0):
            raise ValueError('strictly descending finite alphas from one to zero required')
        for name in ('maximum_mean_fingertip_error_m','minimum_objective_improvement','joint_delta_limit_rad'):
            v=getattr(self,name)
            if not np.isfinite(v) or v<=0:raise ValueError(f'{name} must be finite and positive')


def guarded_joint_selection(original_q,raw_candidate_q,physical_bounds,assess,config=GuardConfig()):
    """Return guarded q and every attempted-alpha assessment.

    ``assess(q)`` must recompute the SAME full-pose objective relative to the
    original native/reference arrays, and the current true mean fingertip
    landmark error. It may compensate the declared palm-anchor convention,
    but must not change object geometry, source pose, weights, sites or scales.
    Scalar objective/mean_fingertip_error_m fields are required. Other audit
    fields are passed through for reproducibility. No scientific gate changes.
    """
    q0=np.asarray(original_q,float).copy();qraw=np.asarray(raw_candidate_q,float).copy()
    limits=np.asarray(physical_bounds,float)
    if (q0.ndim!=1 or len(q0)==0 or qraw.shape!=q0.shape or limits.shape!=(len(q0),2)
        or not np.isfinite(q0).all() or not np.isfinite(qraw).all() or not np.isfinite(limits).all()
        or np.any(limits[:,0]>=limits[:,1])):
        raise ValueError('finite matching joint vectors and Nx2 physical bounds required')
    for q in (q0,qraw):
        if np.any(q<limits[:,0]-1e-9) or np.any(q>limits[:,1]+1e-9):
            raise ValueError('source or raw candidate violates actual physical limits')
    if np.any(np.abs(qraw-q0)>config.joint_delta_limit_rad+1e-9):
        raise ValueError('raw candidate exceeds the frozen joint trust domain')
    def checked(q):
        result=assess(q)
        for name in ('objective','mean_fingertip_error_m'):
            value=result.get(name)
            if value is None or not np.isscalar(value) or not np.isfinite(value) or value<0:
                raise ValueError(f'assessment must provide a nonnegative finite {name}')
        return result
    baseline=checked(q0)
    attempted=[]
    for alpha in config.alphas:
        if alpha==0:
            attempted.append({'alpha':0.,'assessment':baseline,'passed_existing_engineering_guard':
                bool(baseline['mean_fingertip_error_m']<=config.maximum_mean_fingertip_error_m),
                'objective_improved':False,'accepted':False,'reasons':['original_mapping_retained']})
            return q0,{'selected_alpha':0.,'surface_candidate_returned':False,'baseline':baseline,'attempted_alphas':attempted,
                'mapping_fidelity_acceptance':False,'reason':'No frozen interpolant satisfied both engineering conditions.'}
        q=q0+alpha*(qraw-q0)
        result=checked(q)
        guarded=result['mean_fingertip_error_m']<=config.maximum_mean_fingertip_error_m
        improved=result['objective']<baseline['objective']-config.minimum_objective_improvement
        accepted=bool(guarded and improved)
        reasons=[]
        if not guarded:reasons.append('existing_mean_fingertip_error_guard_exceeded')
        if not improved:reasons.append('full_pose_frozen_objective_not_improved')
        attempted.append({'alpha':float(alpha),'assessment':result,'passed_existing_engineering_guard':bool(guarded),
            'objective_improved':bool(improved),'accepted':accepted,'reasons':reasons})
        if accepted:
            return q,{'selected_alpha':float(alpha),'surface_candidate_returned':True,'baseline':baseline,
               'attempted_alphas':attempted,'mapping_fidelity_acceptance':False,
               'reason':'First/largest frozen interpolant satisfies existing fingertip guard and lower full-pose objective.'}
    raise AssertionError('Guard configuration lacks explicit original-target fallback')
