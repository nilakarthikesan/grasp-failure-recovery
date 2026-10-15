"""Pre-execution provenance and manufactured frame-composition review only."""
from pathlib import Path
import ast,hashlib,importlib.util,json
import numpy as np
B=Path(__file__).parent
S=Path('/private/tmp/hug_canonical_mcp_calibration_2026-10-14_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(x):
 global checks
 checks+=1
 if not x:raise AssertionError('Preflight control failed')
checks=0
draft=S/'protocol.draft.json';p=json.loads(draft.read_text());check(sha(draft)=='c51b68e3f2cecced8bff5587357273b75094c6341da327dc863bd08f1701c407')
for table in ('source_sha256','implementation_sha256'):
 for path,h in p[table].items():check(sha(path)==h)
check(not (S/'calibration.json').exists() and not (S/'candidate_targets.json').exists())
for f in ('calibration_math.py','run_calibration.py','test_calibration_math.py'):ast.parse((S/f).read_text())
check(p['supported_NLopt_result_codes']==[1,2,3,4,5,6]);check(p['required_separate_new_anchor_radial_control_before_full_surface_audit'] is True)
spec=importlib.util.spec_from_file_location('independent_calibration_math',S/'calibration_math.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Manufactured positive offset keeps source wrist at zero while C.t is nonzero.
P=np.array([[.03,0,.09],[0,0,.12],[-.03,0,.09]])
R=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
t=R@np.array([0.,0.,.01]);M=P@R.T+t
C,*_=m.calibrate(M,np.zeros(3),P,minimum_longitudinal_m=1e-6,minimum_projected_lateral_m=1e-6)
check(np.max(np.abs(C[:3,:3]-R))<1e-14);check(np.max(np.abs(C[:3,3]-t))<1e-14);check(abs(np.linalg.det(C[:3,:3])-1)<1e-14)
L=np.array([[0,0,0],[.03,.02,.15],[-.02,.04,.11]])
RBP0=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
RWM=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
pWM=np.array([.17,-.09,.3]);RBF=np.diag([-1.,-1.,1.]);pBF=np.array([.03,-.02,.1]);Bpalm=np.array([.01,.02,.15])
vp,ref=m.calibrated_vectors(L,C,RBP0);RWB=RWM@C[:3,:3]@RBP0.T;anchor=pWM+RWM@C[:3,3];tWB=anchor-RWB@Bpalm
RWF=RWB@RBF;tWF=tWB+RWB@pBF;Btask=ref+Bpalm
forearm_task=(Btask-pBF)@RBF
source_world=L@RWM.T+pWM;through_base=Btask@RWB.T+tWB;through_forearm=forearm_task@RWF.T+tWF
check(np.max(np.abs(through_base-source_world))<1e-14);check(np.max(np.abs(through_forearm-source_world))<1e-14)
check(np.linalg.norm(vp[0])>.005)
old_wrong_ref=(vp-vp[0])@RBP0.T
check(np.max(np.abs(old_wrong_ref@RWB.T+anchor-source_world))>.005)
RBPq=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
check(np.max(np.abs(RWF@(RBF.T@RBPq)-RWB@RBPq))<1e-14)
check(np.max(np.abs(RWB@RBPq-RWM@C[:3,:3]))>.5)
for table in ('source_sha256','implementation_sha256'):
 for path,h in p[table].items():check(sha(path)==h)
check(sha(draft)=='c51b68e3f2cecced8bff5587357273b75094c6341da327dc863bd08f1701c407')
check(not (S/'calibration.json').exists() and not (S/'candidate_targets.json').exists())
r={'schema':'hug_canonical_calibration_pre_execution_independent_review_v1','result':'APPROVED_FOR_SEPARATELY_FROZEN_SOURCE_ONLY_CALCULATION','reviewed_draft_sha256':sha(draft),'reviewed_driver_sha256':sha(S/'run_calibration.py'),'review_script_sha256':sha(__file__),'check_count':checks,'source_and_implementation_hashes_verified':100,'actual_source_neutral_model_or_retarget_calculation_run':False,'manufactured_nonzero_palm_offset_world_and_forearm_roundtrip_pass':True,'manufactured_native_wrist_origin_cancellation_detected':True,'q_dependent_palm_rotation_and_fixed_base_consistency_pass':True,'required_unsupported_NLopt_status_refusal_included_before_freeze':True,'new_anchor_radial_control_required_before_full_surface_audit':True,'no_object_or_witness_parameter_tuning':'Manual source/dataflow inspection of canonical(settings), deterministic frame construction and driver; C serialized before observed case loading.','qualified_scope':'Chosen neutral zero-beta canonical template; one new mapping method, not an old frame bugfix. Full physical retarget joint limits; exact original tip caps mandatory. Calibration/numeric success does not establish surface preservation, native failure labels or held-out generalization.','no_production_or_shared_document_edits':True,'new_eval_data':False,'new_inference':False,'physics_steps':0,'training':False}
(B/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
