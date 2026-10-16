"""Independent fixture math; no model, reference layer, search or integration."""
import numpy as np
PART_PARENTS=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
KINDS=('zero_gravity_tracking','authored_gravity_equilibrium','isolated_capsule_plane')
DT=(.002,.001)
DURATIONS=(1.5,1.5,.5)
LIMITS=dict(static_anchor_m=5e-6,static_rotation_component=1e-6,transmission_length=1e-10,zero_servo_force=1e-8,
 tracking_wrist_m=.002,tracking_bone_rad=float(np.deg2rad(2)),gravity_balance_fraction=.01,
 convergence_wrist_m=.001,convergence_bone_rad=float(np.deg2rad(1)),contact_gap_m=1e-9,contact_normal=1e-10,
 contact_J=1e-10,virtual_power_absolute_W=1e-10,virtual_power_relative=1e-10,contact_translation_force_N=1e-8,contact_angular_force_Nm=1e-10,
 cone_primal_relative=1e-8,cone_dual_relative=1e-7,cone_complementarity_relative=1e-7,positive_contact_force_N=1e-8,
 maximum_penetration_m=.005,clock_s=1e-12,raw_ball_norm=1e-8)
FLAGS=dict(accepted_mapping=False,native_task_contact_qualified=False,full_skin_parity_claimed=False,new_collection_or_training_allowed=False)


def finite(a,shape=None):
 a=np.asarray(a,dtype=np.float64)
 if (shape is not None and a.shape!=shape) or not np.isfinite(a).all():raise ValueError('Finite explicitly shaped fixture operand required')
 return a

def quat_product(a,b):
 a=finite(a,(4,));b=finite(b,(4,));return np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]
def quat_exp(r):
 r=finite(r,(3,));angle=np.linalg.norm(r)
 return np.r_[np.cos(angle/2),r*(.5 if angle<1e-15 else np.sin(angle/2)/angle)]
def quat_log(q):
 q=finite(q,(4,))
 if abs(q@q-1)>1e-8:raise ValueError('Unit ball quaternion required')
 q=q if q[0]>=0 else -q;s=np.linalg.norm(q[1:]);return q[1:]*(2. if s<1e-15 else 2*np.arctan2(s,q[0])/s)
def matrix(q):
 w,x,y,z=finite(q,(4,));return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def angle_errors(actual,desired):
 a=finite(actual,(16,3,3));d=finite(desired,(16,3,3));relative=np.einsum('bij,bik->bjk',d,a,optimize=False)
 # atan2 avoids cancellation at the identity; all16, no finger selection.
 s=np.linalg.norm(np.stack((relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]),1),axis=1)/2
 c=(np.trace(relative,axis1=1,axis2=2)-1)/2
 return np.arctan2(s,np.clip(c,-1,1))

def smooth_fraction(time_s,duration):
 x=float(np.clip(time_s/duration,0,1));return x*x*x*(10+x*(-15+6*x))
def trajectory(start,goal,fraction):
 start=finite(start,(67,));goal=finite(goal,(67,));s=float(fraction)
 if not np.isfinite(s) or not 0<=s<=1:raise ValueError('Declared interpolation fraction required')
 out=start.copy();out[:3]=(1-s)*start[:3]+s*goal[:3]
 for j in range(16):
  sl=slice(3+4*j,7+4*j);a=start[sl];b=goal[sl];relative=quat_product(a*np.array([1,-1,-1,-1]),b);out[sl]=quat_product(a,quat_exp(s*quat_log(relative)))
 return out

def fixture_plan():
 return [dict(index=i,kind=kind,dt_s=dt,duration_s=T,steps=int(round(T/dt))) for i in (20,28) for dt in DT for kind,T in zip(KINDS,DURATIONS,strict=True)]

def body_point_J(point,body_index,anchors,rotations):
 p=finite(point,(3,));a=finite(anchors,(16,3));R=finite(rotations,(16,3,3));bid=int(body_index)
 if bid!=body_index or not 0<=bid<16:raise ValueError('Literal native body index required')
 J=np.zeros((3,51));J[:,:3]=np.eye(3)
 while bid>=0:
  J[:,3+3*bid:6+3*bid]=np.cross(R[bid].T,p-a[bid]).T;bid=PART_PARENTS[bid]
 return J

def plane_normal(capsule_rotation):
 R=finite(capsule_rotation,(3,3))
 if np.max(np.abs(R.T@R-np.eye(3)))>1e-10 or abs(np.linalg.det(R)-1)>1e-10:raise ValueError('Proper compiled capsule rotation required')
 return (R[:,2]+R[:,0])/np.sqrt(2)
def plane_quaternion(normal):
 n=finite(normal,(3,))
 if abs(n@n-1)>1e-12:raise ValueError('Unit fixed-plane normal required')
 z=np.array([0.,0.,1.]);cross=np.cross(z,n);dot=z@n
 if dot<-1+1e-12:return np.array([0.,1.,0.,0.])
 q=np.r_[1+dot,cross];return q/np.linalg.norm(q)
def capsule_gap(center,rotation,size,plane_position,normal):
 center=finite(center,(3,));R=finite(rotation,(3,3));size=finite(size,(3,));p=finite(plane_position,(3,));n=finite(normal,(3,))
 return float(n@(center-p)-size[0]-size[1]*abs(n@R[:,2]))

def cone_residual(force,gradient,mu):
 f=finite(force,(3,));g=finite(gradient,(3,))
 if not np.isfinite(mu) or mu<0:raise ValueError('Nonnegative friction required')
 return dict(primal=max(0.,-float(f[0]),float(np.linalg.norm(f[1:])-mu*f[0])),dual=max(0.,float(mu*np.linalg.norm(g[1:])-g[0])),complementarity=abs(float(f@g)),force_scale=float(1+np.linalg.norm(f)),gradient_scale=float(1+np.linalg.norm(g)))

def static_checks(qpos,goal,anchors,rotations,source_anchors,source_rotations,lengths,command,actuator_force):
 checks=dict(qpos=np.array_equal(qpos,goal),anchors=float(np.max(np.abs(finite(anchors,(16,3))-finite(source_anchors,(16,3)))))<=LIMITS['static_anchor_m'],rotations=float(np.max(np.abs(finite(rotations,(16,3,3))-finite(source_rotations,(16,3,3)))))<=LIMITS['static_rotation_component'],transmission=float(np.max(np.abs(finite(lengths,(51,))-finite(command,(51,)))))<=LIMITS['transmission_length'],zero_servo_force=float(np.max(np.abs(finite(actuator_force,(51,)))))<=LIMITS['zero_servo_force'])
 return {k:bool(v) for k,v in checks.items()}

def assess(case,a,goal,source_R,total_mass,wrist_kp):
 t=finite(a['endpoint_time_s']);tail=t>=case['duration_s']-.1-1e-12;checks={};metrics={}
 if len(t)!=case['steps'] or not np.all(tail[-1:]):raise ValueError('Exact declared trajectory required')
 if case['kind']!='isolated_capsule_plane':
  expected=goal[:3].copy()
  if case['kind']=='authored_gravity_equilibrium':expected[2]-=total_mass*9.81/wrist_kp
  wrist=np.linalg.norm(a['endpoint_qpos'][tail,:3]-expected,axis=1);angles=a['bone_angle_error_rad'][tail]
  metrics.update(wrist_tail_max_m=float(wrist.max()),bone_tail_max_rad=float(angles.max()),expected_wrist_world_m=expected.tolist())
  checks['wrist_tracking_or_declared_sag']=metrics['wrist_tail_max_m']<=LIMITS['tracking_wrist_m'];checks['all16_bone_tracking']=metrics['bone_tail_max_rad']<=LIMITS['tracking_bone_rad']
  if case['kind']=='authored_gravity_equilibrium':
   balance=np.mean(a['start_qfrc_actuator'][tail,:3]+a['start_qfrc_passive'][tail,:3]-a['start_qfrc_bias'][tail,:3],axis=0)
   metrics['gravity_tail_force_balance_N']=float(np.linalg.norm(balance));checks['authored_gravity_balance']=metrics['gravity_tail_force_balance_N']<=LIMITS['gravity_balance_fraction']*total_mass*9.81
 else:
  active=a['contact_active'];checks['positive_tail_capsule_reaction']=bool(np.any(active[tail]) and np.mean(a['normal_force_N'][tail])>LIMITS['positive_contact_force_N'])
  for field,limit in [('contact_J_error',LIMITS['contact_J']),('contact_gap_error_m',LIMITS['contact_gap_m']),('contact_normal_error',LIMITS['contact_normal']),('generalized_translation_force_error_N',LIMITS['contact_translation_force_N']),('generalized_angular_force_error_Nm',LIMITS['contact_angular_force_Nm'])]:
   metrics[field]=float(np.max(a[field]));checks[field]=metrics[field]<=limit
  checks['virtual_power']=bool(np.all(a['virtual_power_error_W']<=LIMITS['virtual_power_absolute_W']+LIMITS['virtual_power_relative']*a['virtual_power_scale_W']))
  checks['cone_primal']=bool(np.all(a['cone_primal']<=LIMITS['cone_primal_relative']*a['cone_force_scale']));checks['cone_dual']=bool(np.all(a['cone_dual']<=LIMITS['cone_dual_relative']*a['cone_gradient_scale']));checks['cone_complementarity']=bool(np.all(a['cone_complementarity']<=LIMITS['cone_complementarity_relative']*a['cone_force_scale']*a['cone_gradient_scale']))
 return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,metrics=metrics,physical_outcome=None,**FLAGS)
