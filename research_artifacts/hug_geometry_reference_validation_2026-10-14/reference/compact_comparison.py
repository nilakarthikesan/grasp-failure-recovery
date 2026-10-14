"""Publish an exact compressed trace plus readable geometry report without duplicated traces."""
from pathlib import Path
import copy,gzip,hashlib,json
OUT=Path(__file__).parent
path=OUT/'hug_reference_comparison.json';raw=path.read_bytes();r=json.loads(raw);compact=copy.deepcopy(r)
trace=OUT/'hug_reference_comparison_full.json.gz'
with trace.open('wb') as handle:
 with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0) as compressed:compressed.write(raw)
assert gzip.decompress(trace.read_bytes())==raw
fullsha=hashlib.sha256(raw).hexdigest()
for idx,c in enumerate(compact['cases']):
 for name,m in c['methods'].items():
  if m['status']!='computed':continue
  sol=m['solver'];calls=sol.pop('solver_calls',[]);hist=sol.pop('iteration_history',[])
  if calls:
   sol['solver_call_summary']={'calls':len(calls),'failed_calls':sum(x['exception'] is not None or (x['result_code'] is not None and x['result_code']<0) for x in calls),'return_codes':sorted({x['result_code'] for x in calls if x['result_code'] is not None}),'total_objective_evaluations':sum(x['evaluations'] or 0 for x in calls),'last_call':{k:v for k,v in calls[-1].items() if k not in ('initial_qpos_rad','returned_qpos_rad')}}
  if hist:sol['all_iterations_projected_pair_counts']=[sum(x['projected_intertip_pairs']) for x in hist]
  sol['exact_trace_reference']={'file':trace.name,'uncompressed_full_report_sha256':fullsha,'case_index':idx,'method':name}
compact['compact_presentation']={'full_report_uncompressed_sha256':fullsha,'full_report_gzip_file':trace.name,'omitted_from_compact_only':['Repeated solver initial/returned joint traces and repeated20raw/filtered iteration arrays; exact bytes retained in compressed full report.'],'compaction_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'hug_reference_comparison_compact.json').write_text(json.dumps(compact,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'full_report_sha256':fullsha,'full_gzip_bytes':trace.stat().st_size,'compact_bytes':(OUT/'hug_reference_comparison_compact.json').stat().st_size}))
