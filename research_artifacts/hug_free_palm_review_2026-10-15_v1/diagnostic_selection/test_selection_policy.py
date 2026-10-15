"""Pure synthetic ranking tests, no saved native/frame/geometry inputs."""
from pathlib import Path
import importlib.util,json,hashlib
p=Path(__file__).parent
spec=importlib.util.spec_from_file_location('diagnostic_selector',p/'select_and_verify.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
def row(cost,method=0,seq=1,guarded=False,translated=True,original=False):return {'objective':cost,'method_order':method,'callback_sequence':seq,'is_returned_nonfallback_guarded_pose':guarded,'has_nonzero_world_palm_shift':translated,'is_exact_original_state':original}
checks=[]
def check(v,name):assert v,name;checks.append(name)
a=row(9,guarded=True);b=row(1);chosen,role=s.choose_pool([b,a]);check(chosen is a and role=='guarded_selected','guarded returned pose preferred over refused lower-cost diagnostic')
a=row(3,guarded=True);b=row(2,method=1,guarded=True);check(s.choose_pool([a,b])[0] is b,'lowest comparable returned guarded objective chosen')
a=row(2,method=1);b=row(2,method=0,seq=4);check(s.choose_pool([a,b])[0] is b,'deterministic earlier method tie')
a=row(2,method=0,seq=8);b=row(2,method=0,seq=4);check(s.choose_pool([a,b])[0] is b,'deterministic earlier callback tie')
a=row(5);b=row(0,translated=False,original=True);chosen,role=s.choose_pool([a,b]);check(chosen is a and role=='tip_feasible_translated_diagnostic_not_selected','translated diagnostic chosen even when baseline objective is lower; no silent objective-improvement requirement')
a=row(2,translated=False,original=True);chosen,role=s.choose_pool([a]);check(chosen is a and role=='original_fallback_no_feasible_nonzero','original retained if no translated feasible scored state')
report={'passed':True,'checks':checks,'count':len(checks),'script_sha256':hashlib.sha256((p/'select_and_verify.py').read_bytes()).hexdigest(),'no_native_or_geometry_inputs':True}
(p/'synthetic_selection_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
