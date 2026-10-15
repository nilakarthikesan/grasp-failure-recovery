"""Static translated-palm coordinates and exact-point differential kinematics."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

JOINT_SCALE_RAD = 0.025
TRANSLATION_SCALE_M = 0.005
REACH_OUTER_PADDING_M = 1e-6  # Outer search domain only, never a fingertip acceptance allowance.
TIP_NAMES = ('thtip','fftip','mftip','rftip','lftip')

def decode_state(state, original_q):
    x=np.asarray(state,float);q0=np.asarray(original_q,float)
    if x.shape!=(27,) or q0.shape!=(24,) or not np.isfinite(x).all() or not np.isfinite(q0).all():
        raise ValueError('finite normalized27 state and original24 joints required')
    return q0+JOINT_SCALE_RAD*x[:24], TRANSLATION_SCALE_M*x[24:]

def urdf_tip_reaches(path):
    root=ET.parse(path).getroot();by_child={j.find('child').get('link'):j for j in root.findall('joint')}
    rows=[]
    for tip in TIP_NAMES:
        link=tip;chain=[];seen=set()
        while link!='palm':
            if link in seen or link not in by_child:raise ValueError('tip path fails to terminate at palm')
            seen.add(link);joint=by_child[link]
            if joint.get('type') not in ('revolute','fixed','continuous'):raise ValueError('unexpected nonrotational finger joint')
            origin=joint.find('origin');xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise ValueError('invalid source joint translation')
            length=float(np.linalg.norm(xyz));chain.append({'joint':joint.get('name'),'child':link,'translation_m':xyz.tolist(),'length_m':length})
            link=joint.find('parent').get('link')
        radius=float(sum(c['length_m'] for c in chain))
        rows.append({'tip':tip,'source_chain':chain,'source_reach_upper_m':radius,
                     'outer_domain_padding_m':REACH_OUTER_PADDING_M,'padded_outer_reach_m':radius+REACH_OUTER_PADDING_M})
    return rows

def state_bounds(original_q, physical_limits, native_world_tips, old_caps, original_anchor, reaches):
    q0=np.asarray(original_q,float);limits=np.asarray(physical_limits,float);targets=np.asarray(native_world_tips,float)
    caps=np.asarray(old_caps,float);anchor=np.asarray(original_anchor,float)
    if q0.shape!=(24,) or limits.shape!=(24,2) or targets.shape!=(5,3) or caps.shape!=(5,) or anchor.shape!=(3,):raise ValueError('malformed reach-domain inputs')
    if not all(np.isfinite(v).all() for v in (q0,limits,targets,caps,anchor)) or np.any(caps<0):raise ValueError('invalid reach-domain values')
    rad=np.asarray([r['padded_outer_reach_m'] for r in reaches])+caps
    low_anchor=np.max(targets-rad[:,None],axis=0);high_anchor=np.min(targets+rad[:,None],axis=0)
    if np.any(low_anchor>=high_anchor):raise ValueError('empty conservative translation domain')
    lowq=np.maximum(limits[:,0],q0-.35);highq=np.minimum(limits[:,1],q0+.35)
    qbounds=np.column_stack(((lowq-q0)/JOINT_SCALE_RAD,(highq-q0)/JOINT_SCALE_RAD))
    # Round normalized endpoints inward until decoded q remains inside literal bounds.
    # This only shrinks the search domain; no candidate clamp or acceptance allowance.
    endpoint_shrinks=[]
    for j in range(24):
        counts=[0,0]
        while q0[j]+JOINT_SCALE_RAD*qbounds[j,0] < lowq[j]:
            qbounds[j,0]=np.nextafter(qbounds[j,0],np.inf);counts[0]+=1
        while q0[j]+JOINT_SCALE_RAD*qbounds[j,1] > highq[j]:
            qbounds[j,1]=np.nextafter(qbounds[j,1],-np.inf);counts[1]+=1
        endpoint_shrinks.append(counts)
    bounds=np.concatenate((qbounds,
                           np.column_stack(((low_anchor-anchor)/TRANSLATION_SCALE_M,(high_anchor-anchor)/TRANSLATION_SCALE_M))),axis=0)
    if np.any(bounds[:,0]>0) or np.any(bounds[:,1]<0):raise ValueError('original state outside conservative search domain')
    return bounds,{'tip_order':list(TIP_NAMES),'source_reaches':reaches,'original_caps_m':caps.tolist(),
        'outer_ball_radii_m':rad.tolist(),'native_world_tips_m':targets.tolist(),
        'palm_world_AABB_m':np.column_stack((low_anchor,high_anchor)).tolist(),
        'world_displacement_AABB_m':np.column_stack((low_anchor-anchor,high_anchor-anchor)).tolist(),
        'proof':'For each original-native tip target Ti, ||palm-Ti|| <= source chain reach + original tip cap. Intersection of axis boxes enclosing five balls is a conservative necessary outer domain, not a feasibility certificate.',
        'domain_padding_not_acceptance_allowance':True,'normalized_joint_endpoint_inward_ulp_steps':endpoint_shrinks}

def initial_seeds(bounds):
    result=[];labels=[]
    for distance_m in (.0025,.005,.010):
        for axis in range(3):
            for sign in (1,-1):
                x=np.zeros(27);x[24+axis]=sign*distance_m/TRANSLATION_SCALE_M
                result.append(np.clip(x,bounds[:,0],bounds[:,1]));labels.append(f'translation_{distance_m:g}m_axis{axis}_sign{sign:+d}')
    for sx in (1,-1):
        for sy in (1,-1):
            for sz in (1,-1):
                x=np.zeros(27);x[24:]=np.asarray([sx,sy,sz])/np.sqrt(3)
                result.append(np.clip(x,bounds[:,0],bounds[:,1]));labels.append(f'translation_0.005m_diagonal_{sx:+d}_{sy:+d}_{sz:+d}')
    return np.asarray(result),labels

def virtual_tip_jacobian(model,data,palm_id,tip_positions,tip_body_ids,joint_dof_addresses):
    points=np.asarray(tip_positions,float);bodies=np.asarray(tip_body_ids,int);dofs=np.asarray(joint_dof_addresses,int)
    if points.shape!=(5,3) or bodies.shape!=(5,) or dofs.shape!=(24,):raise ValueError('five actual virtual tip points and24 joint dofs required')
    palm_jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,palm_jac,None,data.xpos[palm_id],palm_id)
    result=np.zeros((5,3,27))
    for i,(point,body) in enumerate(zip(points,bodies,strict=True)):
        tip_jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,tip_jac,None,point,int(body))
        result[i,:,:24]=(tip_jac[:,dofs]-palm_jac[:,dofs])*JOINT_SCALE_RAD
        result[i,:,24:]=np.eye(3)*TRANSLATION_SCALE_M
    return result

def verify_jacobian(state,bounds,errors,analytic,step=1e-4,tolerance_m_per_state=1e-8):
    x=np.asarray(state,float);expected=np.asarray(analytic(x),float);observed=np.zeros_like(expected);spans=[]
    center=errors(x)
    for j in range(27):
        plus=x.copy();minus=x.copy();plus[j]=min(bounds[j,1],x[j]+step);minus[j]=max(bounds[j,0],x[j]-step)
        span=plus[j]-minus[j]
        if span<=0:raise ValueError('nonpositive bounded finite-difference span')
        p=errors(plus) if plus[j]!=x[j] else center;m=errors(minus) if minus[j]!=x[j] else center
        observed[:,:,j]=(p-m)/span;spans.append({'coordinate':j,'minus_delta_scaled':float(minus[j]-x[j]),'plus_delta_scaled':float(plus[j]-x[j]),'span_scaled':float(span)})
    delta=np.abs(expected-observed);errors(x)
    report={'analytic_shape':list(expected.shape),'bounded_finite_difference_step_scaled':step,'spans':spans,
            'analytic_jacobian_m_per_scaled_state':expected.tolist(),'finite_difference_jacobian_m_per_scaled_state':observed.tolist(),
            'maxabs_difference_m_per_scaled_state':float(delta.max()),'threshold_m_per_scaled_state':tolerance_m_per_state,
            'passed':bool(delta.max()<=tolerance_m_per_state),'not_mapping_acceptance':True}
    if not report['passed']:raise ValueError(f'analytic Jacobian gate failed: {report}')
    return report

def stage_translated_palm(model,data,nominal_qpos,joint_qpos_addresses,root_address,palm_id,
                          original_root_position,original_root_quaternion,original_palm_position,
                          original_q,state):
    """Stage the actual floating forearm; compensate wrist FK to requested palm anchor."""
    values,shift=decode_state(state,original_q)
    data.qpos[:]=nominal_qpos;data.qpos[joint_qpos_addresses]=values
    data.qpos[root_address:root_address+3]=original_root_position
    data.qpos[root_address+3:root_address+7]=original_root_quaternion
    mujoco.mj_forward(model,data)
    target=np.asarray(original_palm_position)+shift
    corrected=np.asarray(original_root_position).copy() if np.array_equal(state,np.zeros(27)) else np.asarray(original_root_position)+(target-data.xpos[palm_id])
    data.qpos[root_address:root_address+3]=corrected
    if model.nmocap:data.mocap_pos[0]=corrected;data.mocap_quat[0]=original_root_quaternion
    mujoco.mj_forward(model,data)
    return corrected
