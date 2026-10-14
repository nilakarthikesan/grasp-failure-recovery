"""Meaningful analytic ray checks for the static local-gap diagnostic."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
base=Path(__file__).resolve().parent;source=base/'investigate_geometry.py'
tree=ast.parse(source.read_text());functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('first_ray_entry','nearest_ray_entry')];ns={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),str(source)+':ray-functions-only','exec'),ns)
equations=np.array([[1.,0,0,-1],[-1,0,0,-1],[0,1,0,-1],[0,-1,0,-1],[0,0,1,-1],[0,0,-1,-1]])
box={'id':0,'pos':np.zeros(3),'R':np.eye(3),'equations':equations}
tests=[('entry_to_known_cube',[-2.,0,0],[1.,0,0],1.),('away_from_cube',[-2.,0,0],[-1.,0,0],None),('parallel_outside',[-2.,2,0],[1.,0,0],None),('starting_inside',[0.,0,0],[1.,0,0],0.),('tangent_boundary',[-2.,1,0],[1.,0,0],1.)]
records=[]
for name,q,n,expected in tests:
    actual=ns['first_ray_entry'](box,np.asarray(q),np.asarray(n));assert (actual is None) if expected is None else abs(actual-expected)<1e-12;records.append({'case':name,'actual_entry_m':actual,'expected_entry_m':expected,'passed':True})
# Repeat with a nonidentity proper rotation and translation of the object.
angle=.43;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
trans=np.array([.3,-.4,.2]);transformed={'id':0,'pos':trans,'R':R,'equations':equations};q=np.array([-2.,0,0])@R.T+trans;n=np.array([1.,0,0])@R.T;actual=ns['first_ray_entry'](transformed,q,n);assert abs(actual-1.)<1e-12;records.append({'case':'nonidentity_transform','actual_entry_m':actual,'expected_entry_m':1.,'passed':True})
(base/'ray_sanity.json').write_text(json.dumps({'source_script_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':records,'limitation':'Ray interval intersection includes boundary tangency. Re-entry alone establishes local next-solid boundary, not global cavity connectivity or finite-hand contact access.'},indent=2)+'\n')
