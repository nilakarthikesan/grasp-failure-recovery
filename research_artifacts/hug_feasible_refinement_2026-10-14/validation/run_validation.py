"""Bound runtime controls for the independent static measurement wrapper."""
import os, sys, json, hashlib
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['NUMBA_NUM_THREADS']='4'
os.environ['OMP_DYNAMIC']='FALSE'
os.environ['MKL_DYNAMIC']='FALSE'
OUT=Path(__file__).parent
sys.path.insert(0,str(OUT))
import validate_candidate as runner
import numba
numba.set_num_threads(2)
p=json.loads((OUT/'protocol.json').read_text())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==p['execution_launcher_sha256']
assert numba.config.NUMBA_NUM_THREADS==p['numba_pool_capacity']==4
assert numba.get_num_threads()==p['numba_threads_per_process']==2
print(json.dumps({'runtime_numba_pool':4,'runtime_numba_active_threads':2,'independent_static_audit':True}),flush=True)
runner.main()
assert numba.get_num_threads()==2 and runner.COUNTS['actual']==0
