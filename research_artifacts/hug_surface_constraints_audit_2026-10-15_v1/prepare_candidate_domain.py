"""Add final candidate-domain receipts before the independent audit is frozen."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).parent
assert not (D/'protocol.json').exists()
f=D/'validate_candidate.py';s=f.read_text();before=hashlib.sha256(s.encode()).hexdigest()
anchor="        assert np.linalg.norm(np.asarray(target['target_palm_world_position_m'])-anchor)<1e-12"
block="""        state=np.asarray(target['selected_state_scaled'],float);bounds=np.asarray(target['bounds_scaled'],float);translation_bounds=np.asarray(target['domain']['world_displacement_AABB_m'],float)
        assert state.shape==(27,) and bounds.shape==(27,2) and translation_bounds.shape==(3,2) and np.isfinite(state).all() and np.isfinite(bounds).all() and np.isfinite(translation_bounds).all()
        assert np.all(bounds[:,0]<=bounds[:,1]) and np.all(translation_bounds[:,0]<=translation_bounds[:,1])
        assert np.all(state>=bounds[:,0]) and np.all(state<=bounds[:,1]), 'Frozen primary scaled-domain endpoints, no added allowance'
        assert np.all(shift>=translation_bounds[:,0]) and np.all(shift<=translation_bounds[:,1]), 'Frozen source-reach world AABB, no added allowance'
        assert np.max(np.abs(state[:24]*.025+original_values-qv))<1e-12 and np.max(np.abs(state[24:]*.005-shift))<1e-12
        domain_audit={'scaled_state_and_world_translation_within_declared_frozen_bounds':True,'selected_state_scaled':state.tolist(),'bounds_scaled':bounds.tolist(),'world_displacement_AABB_m':translation_bounds.tolist(),'domain_endpoint_arithmetic_allowance':0.,'state_pose_roundtrip_tolerance':1e-12,'domain_construction_independently_reviewed_separately':True}
"""
assert s.count(anchor)==1;s=s.replace(anchor,block+anchor)
needle="'root_policy':{'translated_anchor_world_m':anchor.tolist()";assert s.count(needle)==1;s=s.replace(needle,"'candidate_domain_audit':domain_audit,"+needle)
ast.parse(s);f.write_text(s)
p=json.loads((D/'protocol.DRAFT.json').read_text());p['script_sha256']=hashlib.sha256(s.encode()).hexdigest();p['candidate_schema']='hug_saved_surface_witness_constraint_candidates_v1';p['candidate_domain_policy']='Final 27D normalized state and actual world palm shift must lie within primary frozen source-reach bounds with zero endpoint allowance; normalized state must reconstruct declared q/shift within1e-12. Independent reviewer checks bound derivation. This domain check is not a scientific surface tolerance.'
(D/'protocol.DRAFT.json').write_text(json.dumps(p,indent=2)+'\n')
par=json.loads((D/'wrapper_parity.json').read_text());par['pre_domain_wrapper_sha256']=before;par['new_script_sha256']=p['script_sha256'];par['amendment']='Surface-constraint selection roles and strict declared primary state/translation-domain check added before execution. Full geometry/status functions and density metric loop remain byte-identical.'
old=Path(par['derived_from']).read_text();a="        for density in selected['densities_mm']:";b="        for file,expected in meta['input_sha256'].items():"
par['full_fields_and_status_byte_identical']=s[:s.index('def main():')]==old[:old.index('def main():')];par['density_metric_block_byte_identical']=s[s.index(a):s.index(b)]==(D/'inherited_metric_block.py.txt').read_text();assert par['full_fields_and_status_byte_identical'] and par['density_metric_block_byte_identical']
(D/'wrapper_parity.json').write_text(json.dumps(par,indent=2)+'\n')
print(json.dumps(par,indent=2))
