"""Read-only evidence verifier; no model load, FK, inference or physics."""
from pathlib import Path
import argparse,gzip,hashlib,json,math,xml.etree.ElementTree as ET

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def report_bytes(root,name):
    plain=root/name
    if plain.is_file():return plain.read_bytes()
    return gzip.decompress((root/(name+'.gz')).read_bytes())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).parent)
    ap.add_argument('--skip-runtime-sources',action='store_true',help='Offline artifact checks only; runtime source checks explicitly unavailable')
    ap.add_argument('--out',type=Path);args=ap.parse_args();out=args.root
    p=read(out/'protocol.json');gp=read(out/'guard_protocol.json');rawb=report_bytes(out,'candidate_targets.json');gb=report_bytes(out,'guarded_candidate_targets.json')
    raw=json.loads(rawb);guarded=json.loads(gb)
    assert raw['protocol_sha256']==sha(out/'protocol.json')
    assert guarded['protocol_sha256']==sha(out/'guard_protocol.json')
    assert hashlib.sha256(rawb).hexdigest()==gp['raw_v2_report_sha256']==guarded['raw_v2_report_sha256']
    assert gp['raw_protocol_sha256']==sha(out/'protocol.json')==guarded['raw_v2_protocol_sha256']
    assert raw['script_sha256']==sha(out/'run_refinement.py') and raw['implementation_sha256']==sha(out/'surface_refinement.py')
    assert guarded['script_sha256']==sha(out/'apply_guard.py') and guarded['guard_implementation_sha256']==sha(out/'guarded_refinement.py')
    for proto in (p,gp):
        for original,h in proto['implementation_sha256'].items():
            assert sha(out/Path(original).name)==h,original
    for r in (raw,guarded):
        assert r['physics_steps']==0 and r['guards']['actual']==0 and r['guards']['selftests']==3
        assert r['new_eval_data'] is False and r['new_generator_proposal'] is False and r['training'] is False
        assert r['protected_hashes_unchanged'] is True
    assert len(raw['cases'])==len(guarded['cases'])==6
    rm={c['native_sha256']:c for c in raw['cases']};runtime_count=0
    if not args.skip_runtime_sources:
        hashes={**p['source_sha256'],**read(out/'protected_hashes.json')}
        for c in raw['cases']:hashes.update(c['input_sha256'])
        for c in guarded['cases']:hashes.update(c['input_sha256'])
        for path,h in hashes.items():assert sha(path)==h,path
        runtime_count=len(hashes)
    ranges=None
    if not args.skip_runtime_sources:
        urdf=next(path for path in p['source_sha256'] if path.endswith('/shadow_hand_right.urdf'))
        ranges={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
          for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    stage_count=0;guard_passes=0
    for c in guarded['cases']:
        previous=rm[c['native_sha256']]
        q0=c['original_qpos_rad'];qraw=c['raw_v2_qpos_rad'];q=c['qpos_rad'];assert len(q0)==len(qraw)==len(q)==24
        assert q0==previous['original_qpos_rad'] and qraw==previous['qpos_rad']
        for values in (q0,qraw,q):assert all(math.isfinite(v) for v in values)
        if ranges is not None:
            for name,original,rawvalue,finalvalue in zip(c['joint_names'],q0,qraw,q,strict=True):
                low,high=ranges[name]
                assert all(low-1e-9<=value<=high+1e-9 for value in (original,rawvalue,finalvalue))
        assert max(abs(a-b) for a,b in zip(qraw,q0))<=p['configuration']['joint_delta_limit_rad']+1e-9
        selection=c['engineering_guard'];alpha=selection['selected_alpha'];attempt=selection['attempted_alphas']
        expected=q0 if alpha==0 else [a+alpha*(b-a) for a,b in zip(q0,qraw)]
        assert expected==q
        assert [a['alpha'] for a in attempt]==gp['guard_configuration']['alphas'][:len(attempt)]
        initial=selection['baseline'];passed=[]
        for a in attempt:
            score=a['assessment'];valid=score['mean_fingertip_error_m']<=gp['guard_configuration']['maximum_mean_fingertip_error_m']
            improved=a['alpha']!=0 and score['objective']<initial['objective']-gp['guard_configuration']['minimum_objective_improvement']
            assert valid==a['passed_existing_engineering_guard'] and improved==a['objective_improved']
            assert (valid and improved)==a['accepted'];passed.append(a['accepted'])
        assert not any(passed[:-1]) and (alpha==0 or passed[-1])
        assert c['final_assessment']['mean_fingertip_error_m']<=gp['guard_configuration']['maximum_mean_fingertip_error_m']
        guard_passes+=1
        stages=[previous['wrist_stage'],*previous['part_diagnostics'].values()]
        for stage in stages:
            assert stage['accepted_objective_improvement'] and stage['solver_success'] and stage['solver_status']>0
            assert stage['after_robust_cost']<stage['before_robust_cost']
            stage_count+=1
    manifest=out/'manifest.json';manifest_count=0
    if manifest.is_file():
        for name,h in read(manifest)['files_sha256'].items():assert sha(out/name)==h,name
        manifest_count=len(read(manifest)['files_sha256'])
    result={'schema':'hug_surface_refinement_artifact_verification_v3','raw_report_exact_uncompressed_sha256':hashlib.sha256(rawb).hexdigest(),
        'guarded_report_exact_uncompressed_sha256':hashlib.sha256(gb).hexdigest(),'operational_protocol_and_implementation_pairs_match':True,
        'all_six_original_and_raw_targets_retained':True,'all_36_stage_objectives_and_solver_status_checked':stage_count==36,
        'guarded_existing_engineering_guard_pass_count':guard_passes,'exact_fixed_alpha_interpolation_and_refusal_trace_checked':True,
        'no_actual_integration_or_generation_recorded':True,'runtime_source_hash_checks_performed':not args.skip_runtime_sources,
        'unique_runtime_source_files_hash_checked':runtime_count,'selected_manifest_files_hash_checked':manifest_count,
        'actual_source_URDF_original_raw_guarded_joint_bounds_checked':ranges is not None,
        'scientific_mapping_acceptance':'NOT_DECIDED_BY_THIS_VERIFIER','new_eval_data':False,'physics_steps':0}
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
