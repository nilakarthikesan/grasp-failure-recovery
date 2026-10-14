"""Known scalar guard cases, fallback/refusal and invalid-domain checks."""
from dataclasses import replace
from pathlib import Path
import hashlib,json
import numpy as np
from guarded_refinement import GuardConfig,guarded_joint_selection

def main():
    q0=np.array([0.]);raw=np.array([.35]);bounds=np.array([[-.5,.5]])
    def score(q):return {'objective':float((q[0]-.2)**2),'mean_fingertip_error_m':float(.002+.03*abs(q[0]))}
    q,r=guarded_joint_selection(q0,raw,bounds,score)
    assert r['selected_alpha']==.5 and np.array_equal(q,np.array([.175]))
    assert [a['alpha'] for a in r['attempted_alphas']]==[1.,.5]
    assert not r['attempted_alphas'][0]['passed_existing_engineering_guard']
    assert r['attempted_alphas'][-1]['accepted']
    def worse(q):return {'objective':float(1+q[0]**2),'mean_fingertip_error_m':.002}
    fallback,rr=guarded_joint_selection(q0,raw,bounds,worse)
    assert np.array_equal(fallback,q0) and rr['selected_alpha']==0 and len(rr['attempted_alphas'])==7
    def refused(q):return {'objective':float((q[0]-.2)**2),'mean_fingertip_error_m':.02}
    poor,rp=guarded_joint_selection(q0,raw,bounds,refused)
    assert np.array_equal(poor,q0) and not rp['surface_candidate_returned']
    rejected=0
    for conf in ({'alphas':(1.,.5,.5,0.)},{'maximum_mean_fingertip_error_m':np.nan},{'joint_delta_limit_rad':0}):
        try:replace(GuardConfig(),**conf)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid guardconfiguration accepted')
    try:guarded_joint_selection(q0,np.array([.36]),bounds,score)
    except ValueError:trust=True
    else:raise AssertionError('Out-of-trust candidateaccepted')
    try:guarded_joint_selection(q0,raw,np.array([[-.1,.1]]),score)
    except ValueError:physical=True
    else:raise AssertionError('Outside physical domainaccepted')
    report={'schema':'hug_guard_manufactured_sanity_v1','largest_allowed_alpha':r['selected_alpha'],
        'selected_joint_rad':q.tolist(),'raw_guard_rejected':True,'lower_alphas_not_unnecessarily_evaluated':True,
        'no_improvement_retains_exact_original':True,'poor_source_not_deleted_or_relabeled':True,
        'invalid_guard_configurations_rejected':rejected,'outside_trust_rejected':trust,'outside_physical_domain_rejected':physical,
        'physics_steps':0,'new_eval_data':False,'guard_implementation_sha256':hashlib.sha256((Path(__file__).parent/'guarded_refinement.py').read_bytes()).hexdigest()}
    (Path(__file__).parent/'guard_manufactured_sanity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
