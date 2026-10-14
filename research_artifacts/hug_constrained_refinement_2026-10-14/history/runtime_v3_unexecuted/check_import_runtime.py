"""Manufactured guarded import smoke; no case/model loading or solver calls."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, os, time

OUT = Path(__file__).parent
V1 = Path('/private/tmp/hug_constrained_refinement_2026-10-14')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def main():
    started = time.perf_counter()
    destination = OUT / 'import_runtime_smoke.json'
    if destination.exists():
        raise FileExistsError('Smoke evidence must not be overwritten')
    parent = read(V1 / 'protocol.json')
    protected = read(OUT / 'protected_hashes.json')
    expected = {**protected, **parent['source_sha256'], **parent['implementation_sha256']}
    assert all(sha(path) == digest for path, digest in expected.items())
    import mujoco
    bootstrap_calls = []

    def forbid(*args, **kwargs):
        bootstrap_calls.append(True)
        raise RuntimeError('Import smoke forbids physics integration')

    for api in ('mj_step', 'mj_step1', 'mj_step2'):
        setattr(mujoco, api, forbid)
    import numba
    assert numba.config.NUMBA_NUM_THREADS == 4
    set_calls = []
    original_set = numba.set_num_threads

    def record_set(value):
        set_calls.append(value)
        return original_set(value)

    numba.set_num_threads = record_set
    source = OUT / 'run_constrained_refinement.py'
    spec = importlib.util.spec_from_file_location('guarded_runtime_import_smoke', source)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert 4 in set_calls and set_calls[-1] == 2
    assert numba.config.NUMBA_NUM_THREADS == 4 and numba.get_num_threads() == 2
    assert driver.COUNTS['actual'] == 0 and not bootstrap_calls
    assert all(getattr(mujoco, api) is driver.forbidden for api in ('mj_step', 'mj_step1', 'mj_step2'))
    driver.COUNTS['testing'] = True
    for api in ('mj_step', 'mj_step1', 'mj_step2'):
        try:
            getattr(mujoco, api)(None, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Import guard selftest failed')
    driver.COUNTS['testing'] = False
    assert driver.COUNTS['actual'] == 0 and driver.COUNTS['selftests'] == 3
    assert all(os.environ.get(variable) == value for variable, value in driver.RUNTIME_ENVIRONMENT.items())
    assert all(sha(path) == digest for path, digest in expected.items())
    numba.set_num_threads = original_set
    report = {
        'schema': 'hug_global_constrained_runtime_v3_guarded_import_smoke',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(__file__), 'driver_sha256': sha(source),
        'numba_set_thread_calls_during_import': set_calls,
        'numba_pool_capacity': numba.config.NUMBA_NUM_THREADS,
        'numba_active_mask_after_import': numba.get_num_threads(),
        'runtime_environment': driver.RUNTIME_ENVIRONMENT,
        'all_three_integration_APIs_bound_to_driver_guard': True,
        'guards': driver.COUNTS, 'bootstrap_guard_actual_calls': len(bootstrap_calls),
        'source_and_protected114_hashes_preserved': True,
        'case_or_model_load_run': False, 'solver_or_candidate_callbacks_run': False,
        'new_eval_data': False, 'new_inference': False, 'physics_steps': 0,
        'runtime_seconds': time.perf_counter() - started,
    }
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
