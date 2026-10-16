"""Hash-bound native task→training readiness bridge. No fitting or admission mint.

Current NativeRecorder/label/trainer versions remain unchanged. Exact external
independent task qualifications are required by the existing loader. Catalog7
coverage is reported separately from the implemented trainer-category5 family.
"""
from pathlib import Path
import argparse,hashlib
import numpy as np
from . import manosim_training as training

SCHEMA='manosim_native_task_readiness_bridge_v1'
CATALOG_SHA='04507e8834364d1451c5c34134d6ca1819efe9da45df5c093d8fa322f897626f'
FOLD_SPEC_SHA='1d7a7549fd75f25d05a1292e3a9d75ff15f8f60ffe88b9d775e93923cb4887a4'


def _raw_binding(row):
 if not isinstance(row,dict) or not {'path','sha256'}<=row.keys() or set(row)-{'path','sha256','bytes'}:raise ValueError('Exact native evidence binding required')
 digest=row['sha256'];digest=digest if digest.startswith('sha256:') else 'sha256:'+digest
 p=training.Binding(path=row['path'],sha256=digest).verify()
 if 'bytes' in row and p.stat().st_size!=row['bytes']:raise ValueError('Bound evidence byte count changed')
 return p


def _coverage(data,assignments):
 if set(assignments)!={a.attempt_id for a in data.schedule.attempts} or not set(assignments.values())<=set(training.ROLES):raise ValueError('All scheduled attempts including exclusions require one role')
 indices={role:np.asarray([i for i,row in enumerate(data.rows) if assignments[row['attempt_id']]==role],dtype=np.int64) for role in training.ROLES}
 scheduled={role:sum(v==role for v in assignments.values()) for role in training.ROLES}
 coverage={role:dict(scheduled=scheduled[role],eligible=len(idx),successes=int(data.y[idx].sum()),future_failures=int(len(idx)-data.y[idx].sum()),groups={key:len({data.rows[i][key] for i in idx}) for key in training.IDENTITIES}) for role,idx in indices.items()}
 # Object/geometry/capture/proposal identity cannot cross roles, even when seen
 # geometry categories intentionally overlap in the object-holdout protocol.
 for key in training.IDENTITIES[1:]:
  groups=[{getattr(a,key) for a in data.schedule.attempts if assignments[a.attempt_id]==role} for role in training.ROLES]
  if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Catalog cross-role grouping leak: '+key)
 ready=all(len(idx)>0 and len(np.unique(data.y[idx]))==2 for idx in indices.values())
 return dict(status='READY' if ready else 'SKIPPED_EMPTY_OR_SINGLE_CLASS_ROLE',coverage=coverage,eligible_attempt_ids={role:[data.rows[i]['attempt_id'] for i in indices[role]] for role in training.ROLES})


def catalog_readiness(data,manifest_path):
 path=Path(manifest_path).resolve();v=training.read(path)
 if (v.get('schema_version')!='manosim_native_pilot_catalog_seven_folds_v1' or v.get('selection_uses_outcomes') is not False
  or v.get('cohort')!='primary120_only_matched1ms_excluded' or v.get('trainer_five_category_manifest_unchanged') is not True or v.get('seven_fold_training_API_implemented') is not False):raise ValueError('Separate frozen catalog7 manifest required')
 catalog=_raw_binding(v['catalog']);spec=_raw_binding(v['source_fold_spec']);files={str(path):training.file_sha256(path),str(catalog):training.file_sha256(catalog),str(spec):training.file_sha256(spec)}
 if files[str(catalog)]!='sha256:'+CATALOG_SHA or files[str(spec)]!='sha256:'+FOLD_SPEC_SHA:raise ValueError('Exact prospective official catalog/fold source required')
 source=training.read(spec);original={f['fold_id']:f for f in source['folds']};folds=v.get('folds',[]);objects={a.object_id for a in data.schedule.attempts}
 if len(original)!=7 or len(folds)!=7 or len({f['fold_id'] for f in folds})!=7 or {f['fold_id'] for f in folds}!=set(original):raise ValueError('Exact seven catalog fold identities required')
 results=[]
 for fold in folds:
  declared=original[fold['fold_id']];roles={role:declared[role+'_object_ids'] for role in training.ROLES};sets={k:set(ids) for k,ids in roles.items()}
  if fold.get('object_membership')!=roles or set.union(*sets.values())!=objects or any(len(sets[k])!=len(roles[k]) for k in sets) or any(sets[a]&sets[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):raise ValueError('Exact disjoint prospective object membership required')
  expected={a.attempt_id:next(role for role,ids in sets.items() if a.object_id in ids) for a in data.schedule.attempts}
  if fold.get('assignments')!=expected or fold.get('generalization_claim')!=declared.get('generalization_claim'):raise ValueError('Outcome-independent catalog role join changed')
  result=_coverage(data,expected)
  if fold['fold_id'].startswith('category_holdout_'):
   category={role:{a.category_id for a in data.schedule.attempts if expected[a.attempt_id]==role} for role in training.ROLES}
   if category['test']&(category['train']|category['validation']):raise ValueError('Held geometry category leaked into training/validation')
  results.append(dict(fold_id=fold['fold_id'],generalization_claim=fold.get('generalization_claim'),training_API_implemented=False,**result))
 return results,files


def _snapshot(data):
 return training.canonical_sha256(dict(bindings=data.bindings,readiness=data.readiness,rows=data.rows,excluded=data.excluded,schedule=data.schedule.model_dump(mode='json'),X_sha256=hashlib.sha256(np.ascontiguousarray(data.X).tobytes()).hexdigest(),y_sha256=hashlib.sha256(np.ascontiguousarray(data.y).tobytes()).hexdigest()))


def export_readiness(collection,category_splits,output,*,qualifications=None,catalog_manifest=None):
 """Attach existing independently qualified evidence, never generate a PASS.

 `qualifications` maps every recording-contract digest to a qualification JSON
 path following manosim_native_task_qualification_v1. Fixture/mechanics-only
 receipts do not satisfy that interface. Missing/invalid evidence stays closed.
 No scientific fitting, optimization, simulation, labeling or feature fitting.
 """
 output=Path(output)
 if output.exists():raise FileExistsError('Fresh readiness output required')
 base=training.read(collection);original_binding=training.binding(collection);base_data=training.load_data(collection)
 candidate=dict(base)
 if qualifications is not None:
  if set(qualifications)!={c.contract_sha256 for c in base_data.schedule.contracts}:raise ValueError('Exact qualification coverage for all native recording contracts required')
  candidate['qualifications']={key:training.binding(path) for key,path in qualifications.items()}
 output.mkdir(parents=True);candidate_path=output/'collection_candidate.json';training.write(candidate_path,candidate)
 data=training.load_data(candidate_path);initial=_snapshot(data);folds=training.split_indices(data,category_splits);split_binding=training.binding(category_splits)
 catalog=[];extra={str(Path(category_splits).resolve()):split_binding['sha256'],str(Path(collection).resolve()):original_binding['sha256']}
 if catalog_manifest is not None:catalog,files=catalog_readiness(data,catalog_manifest);extra.update(files)
 source={str(p):training.file_sha256(p) for p in [Path(__file__).resolve(),*training._source_files()]}
 role_reports=[{k:v for k,v in fold.items() if k!='indices'} for fold in folds]
 ready=bool(data.readiness['scientific_training_allowed'] and any(f['status']=='READY' for f in folds))
 result=dict(schema_version=SCHEMA,scope='native_task_readiness_only',scientific_fits_performed=0,scientific_training_allowed=ready,
  qualification_status='QUALIFIED_NATIVE_TASK_EVIDENCE' if data.readiness['qualified_native_task_evidence'] else 'CLOSED_NO_INDEPENDENT_NATIVE_TASK_QUALIFICATION',
  adequacy_guaranteed=False,collection_candidate=training.binding(candidate_path),original_collection=original_binding,
  implemented_category_training_folds=role_reports,catalog_seven_fold_readiness=catalog,
  catalog_training_API_implemented=False,readiness=data.readiness,eligible_rows=list(data.rows),excluded_rows=list(data.excluded),
  feature_contract=training.feature_contract(training.PrefixSpec()),models=list(training.MODELS),maximum_candidate_fits_per_fold=24,
  model_comparison_policy='same eligible rows and frozen roles for all four models; train-only normalization, validation-only hyperparameters, fixed threshold0.5; seeds are repeats',
  input_bindings={**data.bindings,**extra},source_bindings=source,trainable_collection=None,
  next_action='PREPARE_AND_FREEZE_EXISTING_CATEGORY_TRAINING_PROTOCOL' if ready else 'DO_NOT_FIT; inspect independent task qualification and per-role class/group support')
 # A newly valid but different declaration/data set cannot be substituted while
 # exporting readiness. This follows the training transaction's same guard.
 final=training.load_data(candidate_path)
 if _snapshot(final)!=initial or training.binding(collection)!=original_binding or training.binding(category_splits)!=split_binding:raise ValueError('Native readiness transaction changed')
 if catalog_manifest is not None:
  check,files=catalog_readiness(final,catalog_manifest)
  if check!=catalog or any(result['input_bindings'].get(k)!=v for k,v in files.items()):raise ValueError('Catalog readiness transaction changed')
 for path,digest in {**result['input_bindings'],**source}.items():
  if training.file_sha256(path)!=digest:raise ValueError('Readiness source/input changed')
 if ready:result['trainable_collection']=training.binding(candidate_path)
 training.write(output/'readiness.json',result)
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--category-splits',type=Path,required=True);p.add_argument('--catalog-manifest',type=Path);p.add_argument('--qualifications',type=Path,help='JSON contract digest→existing qualification path; no caller PASS flags');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=export_readiness(a.collection,a.category_splits,a.output,qualifications=training.read(a.qualifications) if a.qualifications else None,catalog_manifest=a.catalog_manifest)
 print(__import__('json').dumps(dict(scientific_training_allowed=result['scientific_training_allowed'],eligible=result['readiness']['eligible'],qualification_status=result['qualification_status'])));return 0
if __name__=='__main__':raise SystemExit(main())
