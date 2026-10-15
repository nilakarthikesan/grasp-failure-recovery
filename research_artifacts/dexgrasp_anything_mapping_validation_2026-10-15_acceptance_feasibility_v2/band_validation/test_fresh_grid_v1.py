from fractions import Fraction
import numpy as np
import validate_band_candidate_v1 as v

def test_fresh_grid_offsets_area_and_covering_radius():
 for spacing,n in [(1.,15000),(.5,60000),(.25,240000)]:
  points,normals,labels,area,rho=v.grid(spacing);h=spacing/1000
  assert len(points)==n and np.isclose(n*area,.015,atol=1e-16,rtol=0)
  assert np.all(np.any(abs(points)==.025,axis=1)) and np.all(np.abs(points)<=.025)
  assert np.all(np.einsum('ij,ij->i',normals,normals)==1.)
  exact_square=Fraction(24,37)**2+Fraction(22,41)**2
  assert abs((rho/h)**2-float(exact_square))<1e-15
  for du in [0.,1.]:
   for dv in [0.,1.]:assert np.hypot(du-13/37,dv-19/41)*h<=rho+1e-18
  assert points[0,1]==-.025+(13/37)*h and points[0,2]==-.025+(19/41)*h
  assert (13/37,19/41) not in [(2/5,1/3),(.5,.5)]

def test_invalid_grid_refused():
 try:v.grid(.3)
 except ValueError:pass
 else:raise AssertionError('Nonpartition grid accepted')
