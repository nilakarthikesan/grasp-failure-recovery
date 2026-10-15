"""Pure transaction-boundary controls; no assets, skin, engine or native query."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('bridge_transaction_controls', Path(__file__).with_name('run_source_pipeline.py'))
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)


def test_receipts_are_write_once_and_reject_nonfinite_json(tmp_path):
    path = tmp_path/'result.json'
    run.write(path, {'status':'HARD_STOP'})
    with pytest.raises(FileExistsError): run.write(path, {'status':'PASS'})
    assert run.read(path) == {'status':'HARD_STOP'}
    with pytest.raises(ValueError): run.write(tmp_path/'nonfinite.json', {'metric':float('nan')})


def test_input_mutation_refuses_without_running_science(tmp_path):
    path = tmp_path/'source'; path.write_text('original')
    bindings = {str(path):run.sha(path)}
    run.verify(bindings); path.write_text('changed')
    with pytest.raises(ValueError): run.verify(bindings)


@pytest.mark.parametrize('change', [
    {'freeze_commit':'replacement'}, {'protocol_sha256':'replacement'}, {'index':28},
    {'schema_version':'replacement'}, {'started_utc':'replacement'},
    {'accepted_mapping':True}, {'native_contact_qualified':True},
    {'new_collection_or_training_allowed':True}, {'physical_outcome':'success'},
    {'forbidden_counts':{'physics_steps':1}}, {'forbidden_counts':None},
    {'status':'PASS'}, {'status':'INCOMPLETE'}])
def test_case_cannot_promote_or_replace_transaction_identity(change):
    measured = dict(status='COMPLETED', forbidden_counts=dict(run.load_case().FORBIDDEN))
    measured.update(change)
    with pytest.raises(ValueError): run.validate_measured(measured)


def test_bounded_completed_measurements_remain_separate_from_admission():
    value = dict(status='COMPLETED', forbidden_counts=dict(run.load_case().FORBIDDEN),
                 bounded_motion_check_passed=False)
    assert run.validate_measured(value) == value


@pytest.mark.parametrize('mutation', ['array', 'raw_receipt', 'tracked_receipt', 'extra_array'])
def test_later_case_cannot_hide_earlier_artifact_mutation(tmp_path, mutation):
    protocol_path = tmp_path/'protocol.json'; run.write(protocol_path, {'declaration':'manufactured'})
    p = dict(protocol_path=str(protocol_path), raw_root=str(tmp_path/'raw'))
    raw = tmp_path/'raw/20'; raw.mkdir(parents=True)
    array = raw/'readback.npz'; array.write_bytes(b'manufactured saved bytes; no geometry')
    artifact = dict(path=str(array), sha256=run.sha(array), bytes=array.stat().st_size)
    progress = raw/'progress.jsonl'; progress.write_text(__import__('json').dumps({'event':'artifact','artifact':artifact})+'\n')
    inventory = [artifact, dict(path=str(progress), sha256=run.sha(progress), bytes=progress.stat().st_size)]
    value = dict(index=20, schema_version=run.SCHEMA+'_case', protocol_sha256=run.sha(protocol_path),
                 freeze_commit='frozen', partial_artifacts=inventory, **run.FLAGS)
    result = raw/'result.json'; tracked = tmp_path/'tracked.json'
    run.write(result, value); tracked.write_bytes(result.read_bytes())
    row = dict(index=20, raw_result=str(result), raw_result_sha256=run.sha(result),
               result=str(tracked), result_sha256=run.sha(tracked))
    run.verify_case_output(row, p, 'frozen')
    if mutation == 'array': array.write_bytes(b'changed earlier result')
    elif mutation == 'raw_receipt': result.write_text('{}')
    elif mutation == 'tracked_receipt': tracked.write_text('{}')
    else: (raw/'unindexed.npz').write_bytes(b'new array')
    with pytest.raises(ValueError): run.verify_case_output(row, p, 'frozen')
