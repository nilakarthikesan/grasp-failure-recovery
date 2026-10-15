"""A later optimizer exception must discard an earlier improving proposal."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import practical_mapper

spec = importlib.util.spec_from_file_location('independent_mapper_fixture', Path(practical_mapper.__file__).with_name('test_practical_mapper.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)

class ContextWithImprovingSeed(fixtures.FakeContext):
    def make_state(self, x, current, block):
        state = super().make_state(x, current, block)
        # Manufactured residual makes q=-.5 better than legal reference q=-1.
        state['errors_m'][1, 0] = abs(state['q']['FFJ4'] + .5)
        return state
    def evaluate_fields(self, state, current=None, block=None):
        fields = np.zeros((856, 8))
        fields[:, 1] = abs(state['q']['FFJ4'] + .5)
        return fields

@pytest.mark.parametrize('exception_type', [ValueError, FloatingPointError, RuntimeError])
def test_exception_after_valid_improving_trial_discards_trial(monkeypatch, exception_type):
    def good_seed(fun, x, **kwargs):
        x = x.copy(); x[0] = -.5
        return SimpleNamespace(x=x, success=True, message='manufactured improvement', nfev=1)
    def failed_solver(*args, **kwargs):
        raise exception_type('manufactured later-stage error')
    monkeypatch.setattr(practical_mapper, 'least_squares', good_seed)
    monkeypatch.setattr(practical_mapper, 'minimize', failed_solver)
    context = ContextWithImprovingSeed()
    baseline, selected, trace = practical_mapper.map_pose(context, np.zeros((856, 8)))
    first = trace['branches'][0]
    assert any(t['strict_improvement'] for t in first['trials'])
    assert selected is baseline
    assert first['optimizer_exception']['type'] == exception_type.__name__
    assert not first['adopted_correction']
    assert context.qsource['FFJ4'] == -2.
