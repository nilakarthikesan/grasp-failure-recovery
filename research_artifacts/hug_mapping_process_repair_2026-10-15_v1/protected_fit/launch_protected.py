"""Preload the exact preflight pose module, then launch the isolated pilot.

The live shared pose_validation.py belongs to concurrent authorized development
and is deliberately not imported or overwritten. The frozen module has no
__file__-based resource lookup; its caller-supplied URDF paths stay unchanged.
"""
from pathlib import Path
import hashlib,importlib,importlib.util,json,sys

HERE=Path(__file__).resolve().parent
INHERITED=Path('/private/tmp/hug_se3_mapping_2026-10-15_v1')
sys.path.append(str(INHERITED))
FROZEN=INHERITED/'frozen_imports/pose_validation.py'
EXPECTED='1136cdfe80c405031e7060c5f31dd3a0270400314e6ff5aa7a4630c487cdea75'
CANONICAL='grasp_failure_prediction.evaluation.pose_validation'

def preload():
    assert hashlib.sha256(FROZEN.read_bytes()).hexdigest()==EXPECTED
    assert CANONICAL not in sys.modules,'Refuse a mixed already-imported pose loader'
    parent=importlib.import_module('grasp_failure_prediction.evaluation')
    assert CANONICAL not in sys.modules,'Parent import unexpectedly loaded live pose module'
    spec=importlib.util.spec_from_file_location(CANONICAL,FROZEN)
    module=importlib.util.module_from_spec(spec)
    sys.modules[CANONICAL]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(CANONICAL,None)
        raise
    setattr(parent,'pose_validation',module)
    assert Path(module.__file__).resolve()==FROZEN.resolve()
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==EXPECTED
    return module

def main():
    module=preload()
    import run_protected
    assert run_protected.base._link_positions is module._link_positions
    assert run_protected.base._body_id is module._body_id
    assert run_protected.base._fixed_tip_offsets is module._fixed_tip_offsets
    assert Path(run_protected.base.a._link_positions.__code__.co_filename).resolve()==FROZEN.resolve()
    expected_urdf=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf')
    assert run_protected.base.a.URDF.resolve()==expected_urdf.resolve()
    assert run_protected.base.numba.get_num_threads()==2
    if sys.argv[1:]==['--verify-imports-only']:
        print(json.dumps({'passed':True,'loaded_pose_path':str(FROZEN),'loaded_pose_sha256':EXPECTED,
            'historical_case_and_auditor_use_frozen_function_objects':True,'default_URDF_unchanged':str(expected_urdf),
            'numba_threads':run_protected.base.numba.get_num_threads(),'scene_instantiations':0,'FK_calls':0,'geometry_queries':0,'physics_steps':0}))
        return
    run_protected.main()

if __name__=='__main__':main()
