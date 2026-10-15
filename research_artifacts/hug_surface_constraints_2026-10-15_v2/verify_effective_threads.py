"""Post-launch import-order audit; no source edits, geometry or optimizer calls."""
from pathlib import Path
import argparse,hashlib,json,os,sys
p=argparse.ArgumentParser();p.add_argument('--method',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.method))
import constraint_engine as e
import numba
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
record={'schema':'hug_numba_import_order_deviation_v1','method_directory':str(a.method),'declared_numba_threads_per_process':2,'effective_numba_threads_after_fresh_identical_engine_import':numba.get_num_threads(),'numba_pool_capacity':numba.config.NUMBA_NUM_THREADS,'engine_sha256':sha(a.method/'constraint_engine.py'),'driver_sha256':sha(a.method/'run_constraints.py'),'nested_frozen_audit_path':e.a.__file__,'nested_frozen_audit_sha256':sha(e.a.__file__),'cause':'constraint_engine sets 2 before importing exact_direction; nested frozen audit module later sets 4. The actual driver does not reset after imports.','launch_inference':'Actual fitting launch commands execute run_constraints.py directly with no post-import reset, so imported effective Numba thread setting is 4 per worker, not protocol-declared 2. No worker timing/speedup claim is made.','scientific_predicates_or_budgets_changed':False,'active_sources_modified':False,'geometry_queries':0,'optimizer_calls':0,'new_eval_data':False}
assert record['effective_numba_threads_after_fresh_identical_engine_import']==4
assert not a.out.exists();a.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
