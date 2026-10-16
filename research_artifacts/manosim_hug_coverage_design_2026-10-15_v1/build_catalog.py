"""Pure metadata join. No model, geometry, pickle loading or dataset execution."""
from pathlib import Path
import hashlib,json,collections
D=Path(__file__).resolve().parent
REVISION='56ea3f100b1a4ec38bb35e986c0dd14d38977283'
SOURCE='https://arxiv.org/html/2606.17054v1#A4.SS2'
CATEGORIES=[('cylindrical',['glue_stick','pepper_shaker','umbrella','bowl','spray_bottle','wine_bottle']),('spheroidal',['strawberry','hacky_sack','pear','softball','pineapple','football']),('prismatic',['eraser','match_box','card_deck','sponge','wipe_dispenser','storage_bin']),('appendaged',['nail_clipper','lock','dustpan','handbell','saucepan','picnic_basket']),('amorphous',['rubber_duck','tape_measure','tape_dispenser','grapes','headphones','easel'])]
def require(value,message):
 if not value:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):(D/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
info=json.loads((D/'1m_hugs_info.json').read_text());tree=json.loads((D/'official_test_tree.json').read_text());require(info['sha']==REVISION,'Pinned dataset revision differs')
files={r['path']:r for r in tree if r['type']=='file'}
def bound(row):
 return dict(dataset_path=row['path'],size_bytes=row['size'],url=f'https://huggingface.co/datasets/kevinywu/1m-hugs/resolve/{REVISION}/'+row['path'],hf_lfs_sha256=row.get('lfs',{}).get('oid'),git_blob_oid=row['oid'])
objects=[]
for category,names in CATEGORIES:
 for j,name in enumerate(names):
  size=('small','medium','large')[j//2];rep=j%2+1;scene=f'{size}_{rep}';oid=f'test/{scene}/{name}';prefix=oid+'/';assets=sorted([bound(r) for p,r in files.items() if p.startswith(prefix+'sim_assets/')],key=lambda r:r['dataset_path']);frames=sorted([bound(r) for p,r in files.items() if p.startswith(prefix) and p.endswith('.pkl')],key=lambda r:r['dataset_path'])
  require(len(frames)==10 and len(assets)>=5 and all(prefix+'sim_assets/'+n in files for n in ('object.xml','object.urdf','visual.obj','properties.json')),'Exact object asset/frame closure missing')
  objects.append(dict(object_id=oid,object_name=name,official_geometry_category=category,official_size_bin=size,official_hug_split='test',scene_id='test/'+scene,cell_replica='A' if rep==1 else 'B',category_and_size_source=SOURCE,category_inferred_from_object_name=False,geometry_asset_files=assets,input_frames=frames,initial_input_frame=frames[0],geometry_closure_bytes=sum(r['size_bytes'] for r in assets),local_admission_status='REMOTE_CATALOG_ONLY_NOT_COMPILED_OR_EXECUTED'))
require(len(objects)==30 and len({o['object_id'] for o in objects})==30,'Exact30object identity required')
cells=collections.Counter((o['official_geometry_category'],o['official_size_bin']) for o in objects);require(len(cells)==15 and set(cells.values())=={2},'Exact15cells/twoobjects required')
catalog=dict(schema_version='official_hug_test_geometry_size_catalog_v1',status='METADATA_VERIFIED_REMOTE_AVAILABLE_NOT_RUNTIME_QUALIFIED',audited_date='2026-10-15',dataset='kevinywu/1m-hugs',dataset_revision=REVISION,dataset_license=info['cardData']['license'],official_table_source=SOURCE,hug_github_revision='8d1c52d4c24bfae5a369e32e3f134f5601a02630',official_hug_bench_archive_revision='25ee2ddd3381a2f5cdf8d64fb577f043a835bfef',objects=objects,summary=dict(object_count=30,geometry_category_count=5,size_bin_count=3,objects_per_cell=2,geometry_files=sum(len(o['geometry_asset_files']) for o in objects),geometry_bytes=sum(o['geometry_closure_bytes'] for o in objects),available_input_frames=300,all_input_frame_bytes=sum(r['size_bytes'] for o in objects for r in o['input_frames']),initial_download_files=sum(len(o['geometry_asset_files'])+1 for o in objects),initial_download_bytes=sum(o['geometry_closure_bytes']+o['initial_input_frame']['size_bytes'] for o in objects)),source_bindings={p.name:sha(p) for p in [D/'1m_hugs_info.json',D/'hug_bench_info.json',D/'hug_github_head.json',D/'official_test_tree.json']},no_model_or_engine_calls=True,no_pickle_execution=True,no_inference_or_fitting=True)
save('official_test_catalog_30.json',catalog)
ids={o['object_id'] for o in objects};folds=[]
def add(fid,kind,train,val,test,details):
 train=set(train);val=set(val);test=set(test);require(train and val and test and not(train&val or train&test or val&test) and train|val|test==ids,'Object-disjoint complete fold required')
 folds.append(dict(fold_id=fid,generalization_claim=kind,train_object_ids=sorted(train),validation_object_ids=sorted(val),test_object_ids=sorted(test),counts=dict(train_objects=len(train),validation_objects=len(val),test_objects=len(test)),details=details))
# Object generalization: all test categories are represented in training. One
# distinct candidate object/category is validation; its size cycles prospectively.
val_sizes=dict(zip([c for c,_ in CATEGORIES],('small','medium','large','small','medium')))
for held in ('A','B'):
 test=[o['object_id'] for o in objects if o['cell_replica']==held];candidate=[o for o in objects if o['cell_replica']!=held];val=[o['object_id'] for o in candidate if o['official_size_bin']==val_sizes[o['official_geometry_category']]];train=[o['object_id'] for o in candidate if o['object_id'] not in val];add('object_holdout_'+held,'UNSEEN_OBJECT_IN_SEEN_GEOMETRY_CATEGORIES',train,val,test,dict(test_one_object_each_geometry_size_cell=True,validation_one_distinct_object_per_category=True,validation_size_by_category=val_sizes,training_has_two_of_three_sizes_per_category=True,limitation='Only10 training and5validation objects/fold; missing one training size cell/category. This is a low-power plumbing pilot.'))
# Category generalization: wholly unseen geometry test. In each remaining cell,
# distinct A/B objects provide balanced train/validation separation.
for category,_ in CATEGORIES:
 test=[o['object_id'] for o in objects if o['official_geometry_category']==category];train=[o['object_id'] for o in objects if o['official_geometry_category']!=category and o['cell_replica']=='A'];val=[o['object_id'] for o in objects if o['official_geometry_category']!=category and o['cell_replica']=='B'];add('category_holdout_'+category,'UNSEEN_GEOMETRY_CATEGORY',train,val,test,dict(test_category_entirely_absent_train_and_validation=True,validation_distinct_objects_in_remaining_categories=True,train_validation_one_object_each_remaining_category_size_cell=True,limitation='12training/12validation/6test objects; results can be class-degenerate and uncertain.'))
spec=dict(schema_version='hug_native_small_pilot_object_category_fold_spec_v1',status='PREPARED_PROSPECTIVE_AWAITING_ROOT_GIT_FREEZE_BEFORE_PROPOSALS',catalog_sha256=sha(D/'official_test_catalog_30.json'),membership_unit='original_physical_object_asset_identity',official_hug_split_retained_separately=True,folds=folds,no_labels_loaded=True,training_performed=False,guardrails=['All source views, seeds, scenarios, perturbations, repeats and trajectory windows of one object stay in its object partition.','Feature normalization fits training rows only; hyperparameters and any early stopping use validation only; threshold0.5 is predeclared.','No test-based winner/cutoff/threshold selection; fixed models and fitting budgets need a separate frozen training protocol.','Class-degenerate folds yield undefined ranking metrics and explicit counts, not fabricated scores.','Different object and category protocols are separate analysis families; do not pool seeds/windows as independent objects.','This new prospective split design does not modify any historical whole-category validation protocol.'])
save('predictor_folds_30.json',spec)
print(json.dumps(catalog['summary'],sort_keys=True));print('catalog',sha(D/'official_test_catalog_30.json'),'folds',sha(D/'predictor_folds_30.json'))
