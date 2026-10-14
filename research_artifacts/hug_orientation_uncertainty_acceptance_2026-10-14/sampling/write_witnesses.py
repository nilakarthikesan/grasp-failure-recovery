"""Auditable static counterexample cells under independently frozen strict rule."""
from pathlib import Path
import hashlib,json
import numpy as np
from cell_intervals import masks
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    protocol=json.loads((OUT/'witness_protocol.json').read_text());ph=sha(OUT/'witness_protocol.json');assert sha(__file__)==protocol['script_sha256'];assert sha(OUT/'acceptance_rule.json')==protocol['acceptance_rule_sha256'];assert sha(OUT/'protocol.json')==protocol['sampling_protocol_sha256']
    p=json.loads((OUT/'protocol.json').read_text());rule=json.loads((OUT/'acceptance_rule.json').read_text());report=json.loads((OUT/'sampling_uncertainty.json').read_text());assert report['protocol_sha256']==protocol['sampling_protocol_sha256'];cache_root=Path('/private/tmp/hug-other-surfaces-2026-10-14');eps=p['primary_allowance_m'];parts=['palm','index','middle','ring','little','thumb','forearm_carrier','wrist_carrier'];required=rule['anatomical_parts'];witnesses=[];results=[]
    for case in report['cases']:
        for density,res in case['resolutions'].items():
            name=f'{case["case_id"]}_density{density}mm';path=OUT/'ephemeral'/f'{name}_full_current.npz';bind=json.loads(path.with_suffix('.binding.json').read_text());assert sha(path)==bind['ephemeral_npz_sha256'] and bind['protocol_sha256']==protocol['sampling_protocol_sha256'];cp=cache_root/'visual_reference_surfaces'/f'{name}_queries.npz';assert sha(cp)==bind['original_cache_sha256']
            with np.load(path,allow_pickle=False) as f:arr={k:f[k].copy() for k in f.files}
            with np.load(cp,allow_pickle=False) as f:nd=f['native_distance_by_part_m'].copy()
            sd=arr['current_distance_by_part_m'];r=arr['cell_radius_m'];area=arr['area_mm2'];q=arr['query_object_m'];record={'case_id':case['case_id'],'density_mm':int(density),'native_sha256':case['native_sha256'],'bands':{}}
            for band in rule['bands_mm']:
                b=band/1000;tests={}
                for part in required:
                    j=parts.index(part);nD,nP=masks(nd[:,j],r,b,eps);mD,mP=masks(sd[:,j],r,b,eps);lost=nD&~mP;new=mD&~nP
                    tests[part]={'lost_lower_area_mm2':float(area[lost].sum()),'new_lower_area_mm2':float(area[new].sum()),'status':'FAIL_STRICT_PRESERVATION_CONDITIONAL' if lost.any() or new.any() else 'UNRESOLVED' ,'numerical_allowance_m':eps}
                    for kind,mask in [('lost_native',lost),('new_mapped',new)]:
                        candidates=np.flatnonzero(mask)
                        if not len(candidates):continue
                        ix=int(candidates[np.argmax(area[candidates])]);dN=float(nd[ix,j]);dM=float(sd[ix,j]);error=float(r[ix]+eps)
                        nlow=max(0,dN-error);nhi=dN+error;mlow=max(0,dM-error);mhi=dM+error
                        assert (nhi<=b and mlow>b) if kind=='lost_native' else (mhi<=b and nlow>b)
                        witnesses.append({'case_id':case['case_id'],'density_mm':int(density),'anatomical_part':part,'band_mm':band,'relationship_change':kind,'child_cell_index':ix,'original_visual_face_index_after_degenerate_exclusion':int(arr['source_face_id'][ix]),'centroid_object_m':q[ix].tolist(),'cell_area_mm2':float(area[ix]),'centroid_radius_m':float(r[ix]),'numerical_allowance_m':eps,'native_centroid_distance_m':dN,'mapped_centroid_distance_m':dM,'native_cell_distance_interval_m':[nlow,nhi],'mapped_cell_distance_interval_m':[mlow,mhi],'native_sha256':case['native_sha256'],'original_query_cache_sha256':bind['original_cache_sha256'],'cell_library_sha256':sha(OUT/'cell_intervals.py'),'sampling_protocol_sha256':protocol['sampling_protocol_sha256'],'acceptance_rule_sha256':protocol['acceptance_rule_sha256']})
                record['bands'][str(band)]=tests
            record['strict_relationship_status']='FAIL_STRICT_PRESERVATION_CONDITIONAL' if any(v['status'].startswith('FAIL') for t in record['bands'].values() for v in t.values()) else 'UNRESOLVED';results.append(record)
    result={'schema':'hug_current_mapping_conditional_strict_counterexamples_v1','witness_protocol_sha256':ph,'script_sha256':sha(__file__),'sampling_report_sha256':sha(OUT/'sampling_uncertainty.json'),'sampling_protocol_sha256':protocol['sampling_protocol_sha256'],'acceptance_rule_sha256':protocol['acceptance_rule_sha256'],'selection':'Largestarea definitecell perrequiredpart/band/relationship/case/density, deterministic earliestindex onarea ties; notworstgeometryselection or calibration.','primary_numerical_allowance_m':eps,'conditioning':'Geometriccellradii arevalidreal-arithmetic coverage; closestquery/coordinatearithmetic uniformerror bound isexplicitassumption, notformalglobalcertificate.','required_anatomical_parts':required,'native_carrier_reference_available':False,'mapped_carriers':'Excludefrom sixpartstrictgate; wholehanddiagnosticincludesallcompatiblecollidersandrecordsitsdistinctdomain.','cases':results,'witness_count':len(witnesses),'witnesses':witnesses,'physics_steps':0,'new_eval_data':False};(OUT/'strict_preservation_witnesses.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'witness_count':len(witnesses),'case_resolution_count':len(results),'report':str(OUT/'strict_preservation_witnesses.json')}))
if __name__=='__main__':main()
