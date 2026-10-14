"""Verify frozen executed inputs, interval containment and witness inequalities."""
from pathlib import Path
import ast,json,hashlib
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=json.loads((OUT/'protocol.json').read_text());r=json.loads((OUT/'sampling_uncertainty.json').read_text());assert r['protocol_sha256']==sha(OUT/'protocol.json');assert r['script_sha256']==sha(OUT/'check_sampling.py')==p['script_sha256'];assert r['cell_library_sha256']==sha(OUT/'cell_intervals.py')==p['cell_library_sha256'];assert r['protected_hashes_unchanged'] and r['guards']['actual']==0 and r['guards']['selftests']==3 and r['physics_steps']==0
    protected=json.loads((OUT/'protected_hashes.json').read_text())
    for file,h in protected.items():assert sha(file)==h
    area_checks=ratio_checks=0
    for case in r['cases']:
        for file,h in case['input_sha256'].items():assert sha(file)==h
        for res in case['resolutions'].values():
            assert res['centroid_and_area_exact_reproduction'] and res['current_cache_band_regression_max_distance_m']<1e-12
            for er in res['numerical_allowances'].values():
                for band in er['bands'].values():
                    for metrics in [band['whole_hand_unsigned_proximity'],*band['independent_parts'].values()]:
                        for key,pointkey in [('native_area','native_area_mm2'),('mapped_area','mapped_area_mm2'),('shared_area','shared_area_mm2'),('union_area','union_area_mm2'),('lost_native_area','lost_native_area_mm2'),('new_mapped_area','new_mapped_area_mm2')]:
                            interval=metrics[key];estimate=metrics['centroid_estimates'][pointkey];assert interval['lower_mm2']-1e-8<=estimate<=interval['upper_mm2']+1e-8;area_checks+=1
                        for key,pointkey in [('iou_interval','iou'),('native_retention_interval','native_retention')]:
                            interval=metrics[key];estimate=metrics['centroid_estimates'][pointkey]
                            if interval['lower'] is not None:assert interval['lower']-1e-12<=estimate<=interval['upper']+1e-12;ratio_checks+=1
    w=json.loads((OUT/'strict_preservation_witnesses.json').read_text());assert w['acceptance_rule_sha256']==sha(OUT/'acceptance_rule.json');assert w['sampling_protocol_sha256']==sha(OUT/'protocol.json');assert w['sampling_report_sha256']==sha(OUT/'sampling_uncertainty.json')
    for cell in w['witnesses']:
        b=cell['band_mm']/1000;nlo,nhi=cell['native_cell_distance_interval_m'];mlo,mhi=cell['mapped_cell_distance_interval_m'];assert cell['cell_area_mm2']>0
        assert (nhi<=b and mlo>b) if cell['relationship_change']=='lost_native' else (mhi<=b and nlo>b)
    for file in OUT.glob('*.py'):ast.parse(file.read_text())
    result={'passed':True,'script_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'source_report_sha256':sha(OUT/'sampling_uncertainty.json'),'area_interval_centroid_containment_checks':area_checks,'ratio_interval_centroid_containment_checks':ratio_checks,'auditable_witness_inequality_checks':len(w['witnesses']),'protected_file_count':len(protected),'protected_hashes_unchanged':True,'all_bound_input_hashes_unchanged':True,'no_actual_physics_calls':True,'all_python_ast_parses':True,'meaning':'Artifact integrity/regression verification, not a formalglobalarithmeticerror or native-contact-equivalencecertificate.'};(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
