import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
geometry=load('test_uncapped_geometry','/private/tmp/dga_surface_preserving_20261015_v1/geometry/surface_backend.py')
prototype=load('test_uncapped_prototype',ROOT/'uncapped_prototype.py')

def test_uncapped_search_ignores_unreferenced_vertex_and_matches_manual_distances():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.1,.1,1.]])
    surface=geometry.TriangleSurface(vertices,np.array([[0,1,2]]))
    query=prototype.UncappedSurface(surface,geometry.closest_triangle_pairs)
    points=np.array([[.1,.1,1.],[1.,1.,0.],[0.,0.,10.]])
    values=query.query(points)
    assert np.allclose(values,[1.,np.sqrt(.5),10.],atol=1e-14,rtol=0)
    assert values[0]>0 # unused near/query vertex must not become surface geometry
    assert values[-1]>.004 # no historical4mm censoring

def test_uncapped_zero_area_triangle_remains_closed_segment():
    surface=geometry.TriangleSurface(np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),np.array([[0,1,2]]))
    query=prototype.UncappedSurface(surface,geometry.closest_triangle_pairs)
    assert np.allclose(query.query(np.array([[.5,1.,0.],[3.,0.,0.]])),[1.,1.],atol=1e-14,rtol=0)

@pytest.mark.parametrize('points',[np.array([[np.nan,0,0]]),np.array([0.,0.,0.])])
def test_uncapped_invalid_queries_fail_closed(points):
    surface=geometry.TriangleSurface(np.eye(3),np.array([[0,1,2]]))
    with pytest.raises(ValueError):prototype.UncappedSurface(surface,geometry.closest_triangle_pairs).query(points)
