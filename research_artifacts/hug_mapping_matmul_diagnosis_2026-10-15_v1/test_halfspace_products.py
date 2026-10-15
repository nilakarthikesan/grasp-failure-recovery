"""Manufactured static numerical/unit tests; no real native/scene execution."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('matmul_operand_replay',Path(__file__).with_name('replay_halfspace_products.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_halfspace_every_output_and_fixed_scalar_samples():
    rng=np.random.default_rng(8)
    normals=rng.normal(size=(13,3));normals/=np.sqrt(np.einsum('ij,ij->i',normals,normals))[:,None]
    result=audit.halfspace_comparison(rng.uniform(-.1,.1,(7,3)),np.column_stack((normals,np.zeros(13))))
    assert result['outputs']==91
    assert result['scalar_sample_count']==16
    assert [r['flat_index'] for r in result['scalar_samples']]==audit.sample_indices(91)
    assert result['max_matmul_einsum_difference_m']<1e-15
    assert result['max_matmul_scalar_difference_m']<1e-15
    assert result['inside_mask_disagreements']==result['hull_predicate_disagreements']==0
    assert result['operands_finite'] and result['outputs_finite']


def test_exact_predicate_boundaries_and_plane_offset():
    result=audit.halfspace_comparison(np.array([[0.,0,0],[1e-10,0,0],[2e-8,0,0]]),np.array([[1.,0,0,0]]))
    assert result['inside_matmul_count']==2
    assert not result['hull_all_vertices_matmul']
    assert result['inside_mask_disagreements']==result['hull_predicate_disagreements']==0
    assert result['scalar_sample_count']==3


@pytest.mark.parametrize('points,planes',[
    ([[float('nan'),0,0]],[[1,0,0,0]]),
    ([[0,0,0]],[[2,0,0,0]]),
    ([[0,0]],[[1,0,0,0]]),
    (np.empty((0,3)),[[1,0,0,0]]),
])
def test_invalid_operands_refused(points,planes):
    with pytest.raises(ValueError):audit.halfspace_comparison(points,planes)


def test_binding_change_rejected(tmp_path):
    source=tmp_path/'source';source.write_text('original')
    bindings={str(source):audit.sha(source)}
    audit.verify(bindings)
    source.write_text('changed')
    with pytest.raises(ValueError,match='changed'):audit.verify(bindings)


def test_warning_counts_preserved():
    import warnings
    def fn():
        warnings.warn('manufactured',RuntimeWarning)
        warnings.warn('manufactured',RuntimeWarning)
        return 2
    value,notices=audit.measured(fn)
    assert value==2
    assert notices==[{'category':'RuntimeWarning','message':'manufactured','count':2}]
