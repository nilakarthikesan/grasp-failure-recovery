"""Independent saved band candidate/field/guard recount. No new field queries."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
R=Path('/private/tmp/dga_mapping_acceptance_20261015_v2');OLD=Path('/private/tmp/dga_component_constraints_20261015_v1');OUT=R/'band_results_v1'
NAMES=('WRJ2','WRJ1','FFJ4','FFJ3','FFJ2','FFJ1','MFJ4','MFJ3','MFJ2','MFJ1','RFJ4','RFJ3','RFJ2','RFJ1','LFJ5','LFJ4','LFJ3','LFJ2','LFJ1','THJ5','THJ4','THJ3','THJ2','THJ1')
FIELDS=('palm','index','middle','ring','little','thumb','wrist_carrier','declared_palm_signed_union')
def sha(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def metrics(s,t):
 assert s.shape==t.shape and s.shape==(856,8) and s.dtype==t.dtype==np.float64
 assert np.isfinite(s).all() and np.isfinite(t).all()
 for x in [s,t]:assert np.all((x[:,:7]>=0)&(x[:,:7]<=.004)) and np.all(np.abs(x[:,7])<=.004)
 result={}
 for k,name in enumerate(FIELDS):
  d=t[:,k]-s[:,k];result[name+'.maximum_abs_difference_m']=float(abs(d).max());result[name+'.MSE_m2']=float(np.mean(d*d))
 sign_cross=np.where(s[:,7]>0,np.maximum(-t[:,7],0),np.where(s[:,7]<0,np.maximum(t[:,7],0),abs(t[:,7])))
 result['declared_palm_occupancy.crossing_depth_max_m']=float(sign_cross.max());result['declared_palm_occupancy.sign_mismatch_count']=int(np.count_nonzero(np.sign(s[:,7])!=np.sign(t[:,7])))
 return result
def main():
 report=read(OUT/'report.json');p=read(OLD/'native_mapping_implementation_frozen_v1.json');frozen=read(R/'control/band_method_frozen_v1.json');g=report['development_global_index'];assert g==frozen['development_global_index']==1
 for path,digest in frozen['dependency_sha256'].items():assert sha(path)==digest
 for name,digest in report['outputs'].items():assert sha(OUT/name)==digest
 old=read(OLD/'native_mapping_results_v1/native_component_fit_report.json');previous=next(x for x in old['records'] if x['global_sample_index']==g)
 ledger=read(OLD/'native_validation/results_v1/native_validation_report.json')['records'][g];algebra=read(OLD/'native_validation/results_v1/native_source_frame_report.json')['records'][g]['transforms']
 with np.load(OUT/'candidate_pose.npz',allow_pickle=False) as z:pose={k:z[k] for k in z.files}
 with np.load(OUT/'fields.npz',allow_pickle=False) as z:fields={k:z[k] for k in z.files}
 with np.load(p['shared_fitting_queries']['path'],allow_pickle=False) as z:assert np.array_equal(z['queries_object_m'],fields['queries_object_m'])
 assert pose['global_sample_indices'].tolist()==[g]
 native=read(ledger['native_manifest']['path']);npz=Path(ledger['native_manifest']['path']).parent/native['output']['path'];assert sha(npz)==native['output']['sha256']
 with np.load(npz,allow_pickle=False) as z:raw=z['hand_qpos'][ledger['within_batch_sample_index']]
 assert pose['source_hand_qpos'][0].dtype==raw.dtype and pose['source_hand_qpos'][0].tobytes()==raw.tobytes()
 assert 'sha256:'+hashlib.sha256(raw.tobytes()).hexdigest()==ledger['raw33_row_sha256']
 initial=metrics(fields['source'],fields['initial']);selected=metrics(fields['source'],fields['selected'])
 assert initial==report['initial_state']['component_metrics']==previous['component_metrics'];assert selected==report['selected_state']['component_metrics'];assert len(initial)==18
 components_nonregressing=all(selected[k]<=initial[k] for k in initial);strict_improvement=any(selected[k]<initial[k] for k in initial);assert components_nonregressing
 adopted=report['adopted_changed_state'];assert type(adopted) is bool
 if adopted:assert strict_improvement
 else:assert np.array_equal(fields['selected'],fields['initial']) and report['selected_state']==report['initial_state']
 q=dict(zip(NAMES,pose['target_joint_values'][0].tolist(),strict=True));qs=dict(zip(NAMES,raw[9:].tolist(),strict=True));state=report['selected_state'];assert q==state['q'];assert q['WRJ2']==qs['WRJ2'] and q['WRJ1']==qs['WRJ1']
 auditpath=p['code_paths']['audit'];spec=importlib.util.spec_from_file_location('independent_band_fk',auditpath);a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
 source=a.model(Path(p['source_hand_urdf']));target=a.model(Path(p['target_hand_urdf']));sf=a.independent_fk(source,qs);tf=a.independent_fk(target,q)
 for name,value in q.items():
  lim=target['joints'][name]['limit'];assert lim['lower']<=value<=lim['upper']
 for name,value in qs.items():
  lim=source['joints'][name]['limit'];assert lim['lower']<=value<=lim['upper']
 source_root=pose['object_from_source_forearm'][0];source_palm=pose['object_from_source_palm'][0];root=pose['object_from_target_root_world'][0];palm=pose['object_from_target_palm'][0];forearm=pose['object_from_target_forearm'][0];delta=pose['source_palm_from_target_palm'][0]
 residuals={'source_forearm_binding':float(np.max(abs(source_root-np.array(algebra['object_from_source_forearm'])))),'source_palm_binding':float(np.max(abs(source_palm-np.array(algebra['object_from_palm'])))),'source_palm_chain':float(np.max(abs(source_root@sf['palm']-source_palm))),'target_palm_chain':float(np.max(abs(root@tf['palm']-palm))),'target_forearm_chain':float(np.max(abs(root@tf['forearm']-forearm))),'palm_delta_chain':float(np.max(abs(source_palm@delta-palm)))}
 assert max(residuals.values())<=1e-9
 for name,t in [('root',root),('palm',palm),('forearm',forearm),('delta',delta)]:
  assert np.array_equal(t,np.array(state[name]));assert np.array_equal(t[3],[0,0,0,1]);assert np.max(abs(t[:3,:3].T@t[:3,:3]-np.eye(3)))<=1e-9 and abs(np.linalg.det(t[:3,:3])-1)<=1e-9
 fingers=[n for n in NAMES if not n.startswith('WRJ')];assert np.array_equal(np.array(state['variables'])[:22],[q[n] for n in fingers])
 sites=['palm']+sorted(n for n in source['links'] if n in target['links'] and n.startswith(('ff','mf','rf','lf','th')));assert len(sites)==28
 errors=np.stack([(root@tf[n])[:3,3]-(source_root@sf[n])[:3,3] for n in sites]);norm=np.sqrt(np.einsum('ij,ij->i',errors,errors));independent={'maximum_site_error_m':float(norm.max()),'mean_site_error_m':float(norm.mean()),'all_site_SSE_m2':float(np.sum(errors*errors))};caps=report['landmark_caps'];assert caps==previous['landmark_limits']
 landmarks_pass={k:independent[k]<=caps[k] for k in caps};difference={k:independent[k]-state['landmark_metrics'][k] for k in caps};site_difference=float(np.max(abs(norm-np.array(state['landmark_metrics']['per_site_error_m']))));assert max(abs(x) for x in difference.values())<1e-12 and site_difference<1e-12
 result={'schema_version':'dga_band_saved_candidate_independent_recount_v1','status':'PASS_ALL_SAVED_GUARDS_RECOUNTED' if all(landmarks_pass.values()) else 'INDEPENDENT_ARITHMETIC_BOUNDARY_REQUIRES_REVIEW','adopted_changed_state':adopted,'exact_native_bytes_and_original_wrists':True,'strict_named_target_limits':True,'component_count':18,'componentwise_nonregression':components_nonregressing,'strict_component_improvement':strict_improvement,'independent_landmark_metrics':independent,'unchanged_landmark_caps':caps,'independent_landmark_pass_by_cap':landmarks_pass,'independent_vs_saved_landmark_difference':difference,'max_per_site_difference_m':site_difference,'frame_residuals_max_entry':residuals,'frozen_fitting_queries_exact':True,'case_global_index':g,'reviewed_files':{str(x):sha(x) for x in [Path(__file__),OUT/'report.json',OUT/'candidate_pose.npz',OUT/'fields.npz',OUT/'trial_trace.json',R/'control/band_method_frozen_v1.json']},'scientific_acceptance':'NOT_GRANTED','execution_eligible':False,'fresh_surface_queries':0,'physics_steps':0}
 out=R/'contract_review/band_saved_candidate_independent_recount_v1.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':result['status'],'adopted_changed_state':adopted,'sha256':sha(out)}))
if __name__=='__main__':main()
