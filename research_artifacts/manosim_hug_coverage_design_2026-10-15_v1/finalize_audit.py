"""Saved metadata/hash audit only; never load pickles, meshes or models."""
from pathlib import Path
import collections,hashlib,json
D=Path(__file__).resolve().parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
BASE=Path('/Users/nilakarthikesan/grasp-failure-prediction')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x): (D/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def require(x,msg):
 if not x: raise ValueError(msg)
c=json.loads((D/'official_test_catalog_30.json').read_text());f=json.loads((D/'predictor_folds_30.json').read_text());tree=json.loads((D/'official_test_tree.json').read_text());by={o['object_id']:o for o in c['objects']};ids=set(by)
require(sha(D/'official_test_catalog_30.json')=='04507e8834364d1451c5c34134d6ca1819efe9da45df5c093d8fa322f897626f','Catalog preimage changed')
require(sha(D/'predictor_folds_30.json')=='1d7a7549fd75f25d05a1292e3a9d75ff15f8f60ffe88b9d775e93923cb4887a4','Fold preimage changed')
checks=[]
for fold in f['folds']:
 tr,va,te=[set(fold[k]) for k in ('train_object_ids','validation_object_ids','test_object_ids')]
 require(not(tr&va or tr&te or va&te) and tr|va|te==ids,'Object overlap/incomplete partition')
 if fold['fold_id'].startswith('object_'):
  require((len(tr),len(va),len(te))==(10,5,15),'Object fold counts')
  require(len({by[i]['official_geometry_category'] for i in tr})==5,'Missing train category')
  require(len({(by[i]['official_geometry_category'],by[i]['official_size_bin']) for i in te})==15,'Test cell coverage')
 else:
  require((len(tr),len(va),len(te))==(12,12,6),'Category fold counts')
  require(not({by[i]['official_geometry_category'] for i in te}&{by[i]['official_geometry_category'] for i in tr|va}),'Category leakage')
 checks.append(dict(fold_id=fold['fold_id'],counts=fold['counts'],object_disjoint=True,partition_complete=True))
assetroot=ROOT/'runs/hug_native_pilot_assets_2026-10-15_v1';mp=assetroot/'download_manifest.json';m=json.loads(mp.read_text());remote={r['path']:r for r in tree if r['type']=='file'}
expected={a['dataset_path'] for o in c['objects'] for a in o['geometry_asset_files']}
expected|={o['input_frames'][j]['dataset_path'] for o in c['objects'] for j in (0,-1)}
require({r['path'] for r in m['files']}==expected,'Downloaded selection differs from entire fixed30 matrix+first/last')
for row in m['files']:
 p=assetroot/row['path'];r=remote[row['path']];data=p.read_bytes()
 require(len(data)==row['bytes']==r['size'] and hashlib.sha256(data).hexdigest()==row['sha256'],'Downloaded byte mismatch')
 if r.get('lfs'):
  require(row['sha256']==r['lfs']['oid'],'Remote LFS content mismatch')
 else:
  require(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==r['oid'],'Remote Git blob content mismatch')
require(len(m['files'])==442 and sum(r['bytes'] for r in m['files'])==18417075,'Download totals differ')
local=[]
for mp0 in [BASE/'runs/hug_benchmark_review_2026-10-09/download_manifest.json',*[BASE/f'runs/hug_build_heldout_assets_2026-10-15_v1/{n}/download_manifest.json' for n in ('bell_pepper','coffee_cup','bear_figure')]]:
 if not mp0.exists():continue
 z=json.loads(mp0.read_text());verified=0;unresolved=[]
 for r in z.get('files',[]):
  raw=r.get('local_path',r.get('path'))
  if raw is None:unresolved.append('missing path');continue
  p=Path(raw)
  if not p.is_absolute():p=mp0.parent/p
  if not p.exists():unresolved.append(str(p));continue
  if r.get('sha256')!=sha(p):raise ValueError('Historical local file changed '+str(p))
  verified+=1
 local.append(dict(manifest=str(mp0),sha256=sha(mp0),verified_file_hashes=verified,unresolved_paths=unresolved,not_loaded_or_executed=True))
sourceaudit=dict(schema_version='hug_native_coverage_source_audit_v1',status='VERIFIED_METADATA_AND_DOWNLOADED_BYTES_ONLY',audited_date='2026-10-15',catalog_sha256=sha(D/'official_test_catalog_30.json'),fold_spec_sha256=sha(D/'predictor_folds_30.json'),dataset_revision=c['dataset_revision'],download_manifest=dict(path=str(mp),sha256=sha(mp),files=442,bytes=18417075,objects=30,observations=60,selection='all source-listed geometry files plus first and last lexicographic frame per object',every_file_sha256=True,every_file_remote_git_or_lfs_identity=True),fold_checks=checks,historical_local_manifests=local,operations=dict(model_calls=0,engine_calls=0,inference_calls=0,pickle_loads=0,geometry_queries=0,training_calls=0),limitations=['Byte availability is not geometric, dynamical, controller or task-label qualification.','Official size bins are the authors labels; no metric cutoffs inferred and no mesh scaling performed.','Two stored observations per object are views; they are not evidence of two independent human grasps.','All30 assets belong to the official HUG test split. Predictor splits below are separate prospective partitions, not a change to official HUG training.'])
save('source_audit.json',sourceaudit)
rationale=dict(schema_version='hug_native_coverage_rationale_v1',status='PROSPECTIVE_SYSTEM_PILOT_DESIGN_NO_TASK_RESULTS',sole_generator='HUG',embodiment='native MANO in MANOSim',purpose='First validate faithful generated target transfer and a runnable controller/physics pipeline, then collect measured failures for pre-lift prediction.',precedents=[dict(title='Human Universal Grasping',authors='Kevin Yuanbo Wu et al.',year=2026,paper_version='2606.17054v1;15 June2026',url='https://arxiv.org/html/2606.17054v1',sections=['5.1','D.2 Table6'],borrow='Use the published five geometry categories by three size bins, with two official test objects per cell. Select all30 independently of reported success.',limits='Author-defined geometric categories and size bins are retained. This catalog does not claim new performance or universal metric size cutoffs.'),dict(title='UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy',authors='Yinzhen Xu et al.',year=2023,paper_version='2303.00938v2;25 March2023',url='https://arxiv.org/html/2303.00938v2',sections=['3.2','4.1'],facts='The proposal formulation separates orientation from translation/articulation. Its object split separates3251training objects,754unseen instances in seen categories, and1514objects in unseen categories.',borrow='Track pose variety separately from object/category generalization. Group all views, proposals and scenarios of one object together.',limits='Do not add another generator or copy a robot embodiment/policy. Many seeds on one object do not demonstrate new-object transfer.'),dict(title='Dex-Net2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics',authors='Jeffrey Mahler et al.',year=2017,paper_version='1703.09312v3;8 August2017',url='https://arxiv.org/html/1703.09312v3',sections=['IV-A1 and TableI','VI-B'],borrow='Separate physical variation such as pose/friction from an observation-noise model. Preserve reliable labels while changing physical scenarios or measurements explicitly.',limits='Its robust analytic grasp-quality/collision labels are not measured MANO lift/hold outcomes. Its image-wise80/20split is not our object-generalization split; do not add random label errors to create failures.')],pilot=dict(objects=30,geometry_categories=5,size_bins=3,objects_per_cell=2,views_per_object=2,grasp_seeds_per_view=2,proposed_primary_native_requests=120,repeat_budget='Separately declared by frozen inference protocol; repeat rows do not increase independent coverage.',seeds_and_actual_generation='Owned by root generation_protocol.json; no proposals generated by this audit.',selection='Entire official test matrix, first/last lexicographic observation, before new proposals/outcomes.'),fold_rationale=dict(object_holdouts='Two reciprocal outer folds:15test objects, one per cell; among15remaining,5distinct validation objects, one per category at predeclared size, and10train. All categories appear in train, but one size/cell per category is reserved for validation.',category_holdouts='Five outer folds: all6objects in one geometry category test;12A objects from other categories train and12distinctB objects validation. The held category appears in neither train nor validation.',grouping='All views/seeds/scenarios/repeats/trajectory windows of a physical object remain in its partition.',limitations=['These small folds are for pipeline coverage and tentative estimates, not evidence that a predictor is adequately trained.','Category folds share scene/background identities across different objects in remaining and held categories; they support a geometry holdout claim, not an unseen-camera/scene claim.','Whole-scene purging would empty training for this matrix because the held category spans every scene.','State-prefix inputs exclude image/background/asset/category identity and future labels; nevertheless scene correlation must be disclosed.','A new prospective split does not overwrite earlier historical whole-category-validation protocols.']),future_collection_contract=['Freeze controller/profile/scorer/feature cutoff and budgets after gravity/approach/multifinger qualification and before task outcomes.','Compare pre-lift state/command prefixes only. Future lift/drop/hold events provide labels, never input features.','Approach failures preceding cutoff belong to an already-failed risk set, not prospective warning successes. Mapping/setup/numerical/runtime refusals retain null physical labels.','Use measured success/failure definitions and retain failed proposals. Do not use analytic quality or generator score as task ground truth.','Declare physical-randomization blocks separately from observation noise, and audit actual effective contact parameters.','MuJoCo higher-priority geom controls contact friction; equal priority uses elementwise maximum. Object friction alone may have no effect if hand friction dominates. Verify actual contact friction/priority or qualify a separately declared pair/both-geom intervention.','Training-only normalization, validation-only tuning, fixed0.5threshold, unchanged cutoff and same held rows for model comparisons; class-degenerate folds explicitly undefined.'],effective_friction_source='https://mujoco.readthedocs.io/en/3.4.0/modeling.html#contact-parameters',runtime_admission=False,training_performed=False,accuracy_guarantee=False)
save('coverage_rationale.json',rationale)
print(json.dumps(dict(status=sourceaudit['status'],catalog_sha256=sha(D/'official_test_catalog_30.json'),fold_sha256=sha(D/'predictor_folds_30.json'),download_files=442,download_bytes=18417075,historical=local),indent=2))
