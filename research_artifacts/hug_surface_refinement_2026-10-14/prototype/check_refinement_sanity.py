"""Manufactured static callback and contract checks; no MuJoCo or episodes."""
from dataclasses import replace
from pathlib import Path
import hashlib,json
import numpy as np
from surface_refinement import RefinementConfig,area_quantile_sites,select_sites,refine_part

def main():
    invalid=[{'distance_scale_m':0},{'joint_delta_limit_rad':np.inf},{'near_mass':1},
             {'near_sites':0},{'max_nfev':2.5},{'loss':'linear'},{'closest_vector_weight':-1}]
    count=0
    for update in invalid:
        try:replace(RefinementConfig(),**update)
        except ValueError:count+=1
        else:raise AssertionError(f'Config accepted {update}')
    areas=np.array([1.,3.,2.]);mask=np.array([True,True,False])
    chosen=area_quantile_sites(mask,areas,4)
    assert np.array_equal(chosen,[0,1,1,1])
    q0=np.array([.2]);bounds=np.array([[-.5,.5]])
    nd=np.array([.1]);ncp=np.array([[0.,0.,.1]])
    facing=np.array([-1.]);eligible=np.array([False]);weights=np.array([1.]);anchors=np.zeros((1,3))
    def plane(q):
        # A translated infinite plane is the exact local static analogue.
        return np.array([abs(q[0])]),np.array([[0.,0.,q[0]]]),np.array([-1.]),anchors
    target,report=refine_part(q0,bounds,nd,ncp,facing,eligible,weights,anchors,plane)
    assert report['accepted_objective_improvement']
    assert abs(target[0]-.1)<.0002
    assert np.max(np.abs(target-q0))<=.35
    assert report['after_robust_cost']<report['before_robust_cost']
    def invalid_shape(q):return np.array([.1]),np.zeros((2,3)),np.array([0.]),anchors
    try:refine_part(q0,bounds,nd,ncp,facing,eligible,weights,anchors,invalid_shape)
    except ValueError:shape_rejected=True
    else:raise AssertionError('Malformed callback accepted')
    try:area_quantile_sites(mask,np.array([1.,np.nan,2.]),4)
    except ValueError:areas_rejected=True
    else:raise AssertionError('Nonfinite areas accepted')
    try:refine_part(q0,np.array([[-.1,.1]]),nd,ncp,facing,eligible,weights,anchors,plane)
    except ValueError:physical_rejected=True
    else:raise AssertionError('Physical limit violation accepted')
    result={'schema':'hug_static_refinement_manufactured_sanity_v1','invalid_configuration_count':count,
       'deterministic_area_quadrature_indices':chosen.tolist(),'quadrature_duplicates_are_not_independent':True,
       'known_plane_target_qpos_rad':.1,'returned_qpos_rad':target.tolist(),'absolute_known_plane_error_rad':abs(float(target[0])-.1),
       'plane_objective_before':report['before_robust_cost'],'plane_objective_after':report['after_robust_cost'],
       'malformed_callback_rejected':shape_rejected,'nonfinite_area_rejected':areas_rejected,
       'actual_physical_limit_violation_rejected':physical_rejected,'native_nonfacing_state_not_replaced':True,
       'new_eval_data':False,'physics_steps':0,'implementation_sha256':hashlib.sha256((Path(__file__).parent/'surface_refinement.py').read_bytes()).hexdigest()}
    (Path(__file__).parent/'manufactured_sanity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
