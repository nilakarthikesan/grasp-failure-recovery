"""Manufactured nonidentity collider refresh vs fresh compiled geometry."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from geometry_helpers import refresh_colliders

CALLS={'actual':0}
def blocked(*args,**kwargs):
    CALLS['actual']+=1
    raise RuntimeError('Manufactured rigid-refresh check forbids integration')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,blocked)
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
LIB=ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py'
sp=importlib.util.spec_from_file_location('frozen_geometry_for_refresh',LIB);a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,blocked)
import numba
numba.set_num_threads(1)

def main():
    xml='''<mujoco><compiler angle="radian"/><asset><mesh name="tetra"
      vertex="0 0 0  .012 0 0  0 .009 0  0 0 .015"/></asset>
      <worldbody><body name="carrier"><freejoint/><geom type="mesh" mesh="tetra"
      pos=".004 -.006 .002" euler=".17 -.21 .31"/>
      <geom type="box" size=".006 .004 .008" pos=".015 .008 -.003" euler="-.27 .35 -.13"/>
      <geom type="sphere" size=".006" pos="-.015 .007 .003"/>
      <geom type="cylinder" size=".004 .010" pos=".007 -.018 .004" euler=".2 .1 -.4"/>
      </body></worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    objR=Rotation.from_euler('xyz',[-.41,.29,-.36]).as_matrix();objpos=np.array([.021,-.017,.038])
    original=[a.geom_rep(m,d,g,objR,objpos) for g in range(m.ngeom)]
    saved_source_arrays=[{k:v.copy() for k,v in g.items() if isinstance(v,np.ndarray)} for g in original]
    old_world_tri=[None if g['tri_object'] is None else g['tri_object'].copy() for g in original]
    R=Rotation.from_euler('xyz',[.37,-.23,.62]).as_matrix();quat=np.empty(4);mujoco.mju_mat2Quat(quat,R.reshape(9))
    d.qpos[:3]=[.031,-.025,.044];d.qpos[3:7]=quat;mujoco.mj_forward(m,d)
    refreshed=refresh_colliders(original,m,d,objpos,objR)
    fresh=[a.geom_rep(m,d,g,objR,objpos) for g in range(m.ngeom)]
    points=np.array([[.002,.007,.003],[.015,-.008,.026],[-.023,.012,.045],
        [.018,.031,-.017],[.052,-.016,.009],[-.019,-.037,.024]])
    records=[]
    for source,updated,independent,oldtri,saved in zip(original,refreshed,fresh,old_world_tri,saved_source_arrays,strict=True):
        for name,value in saved.items():assert np.array_equal(source[name],value),name
        position_error=float(np.max(np.abs(updated['pos']-independent['pos'])))
        rotation_error=float(np.max(np.abs(updated['R']-independent['R'])))
        assert position_error<1e-15 and rotation_error<1e-15
        triangle_error=None;stale_shift=None
        if updated['tri_object'] is not None:
            triangle_error=float(np.max(np.abs(updated['tri_object']-independent['tri_object'])))
            stale_shift=float(np.max(np.abs(oldtri-independent['tri_object'])))
            assert triangle_error<1e-15 and stale_shift>.001
            assert np.array_equal(source['tri_object'],oldtri)
        ud,uc,un=a.geom_query(updated,points);fd,fc,fn=a.geom_query(independent,points)
        errors={'signed_distance_max_error_m':float(np.max(np.abs(ud-fd))),
            'closest_point_max_error_m':float(np.max(np.abs(uc-fc))),
            'normal_max_error':float(np.max(np.abs(un-fn)))}
        assert max(errors.values())<1e-14
        records.append({'geom_id':source['id'],'kind':source['kind'],'position_max_error_m':position_error,
            'rotation_max_error':rotation_error,'object_frame_triangle_max_error_m':triangle_error,
            'old_stale_triangle_change_m':stale_shift,'local_immutable_geometry_preserved':True,**errors})
    assert d.time==0 and CALLS['actual']==0
    report={'schema':'hug_constrained_rigid_refresh_manufactured_v1','geometry_library_sha256':a.sha(LIB),
        'refresh_implementation_sha256':a.sha(Path(__file__).parent/'geometry_helpers.py'),
        'nonidentity_hand_and_object_frames':True,'fresh_compiled_geom_rep_parity':True,
        'geom_count':m.ngeom,'cases':records,'time_s':float(d.time),'actual_integration_attempts':0,
        'new_eval_data':False,'real_saved_candidate_metrics':False,'licensed_assets_loaded':False}
    (Path(__file__).parent/'rigid_refresh_sanity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
