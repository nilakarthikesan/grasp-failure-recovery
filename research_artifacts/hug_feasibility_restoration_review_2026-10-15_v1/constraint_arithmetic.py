"""Independent replay of saved witness arithmetic. No primary core import."""
import numpy as np

class Replay:
    def __init__(self,whole_rows,orientation_rows,index,protocol,caps):
        self.p=[r for r in whole_rows if r['source_inventory_index']==index]
        self.o=[r for r in orientation_rows if r['source_inventory_index']==index]
        self.protocol=protocol;self.caps=np.asarray(caps,float);self.checks=0
        self.near=np.array([r['row']['relationship_change']=='lost_native' for r in self.p],bool)
        self.radius=np.array([r['row']['centroid_radius_m'] for r in self.p],float)
        self.allowance=np.array([r['row']['numerical_allowance_m'] for r in self.p],float)
        self.band=np.array([r['row']['band_mm']/1000 for r in self.p],float)
        self.sign=np.sign([r['row']['native_facing_compatibility'] for r in self.o])
        self.oband=np.array([r['band_mm']/1000 for r in self.o],float)
        self.on=np.array([r['row']['object_outward_normal'] for r in self.o],float).reshape(-1,3)
        parts=['palm','index','middle','ring','little','thumb','forearm_carrier','wrist_carrier']
        self.part=np.array([parts.index(r['row']['native_part']) for r in self.o],int)
    def check(self,v,label):
        self.checks+=1
        if not bool(v):raise AssertionError(label)
    def equal(self,a,b,label):self.check(np.array_equal(np.asarray(a),np.asarray(b)),label)
    def close(self,a,b,label,tol=1e-14):
        a=np.asarray(a);b=np.asarray(b)
        self.check(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' shape/finite')
        self.check(np.max(np.abs(a-b),initial=0)<=tol,label)
    def orientation(self,score,distance,valid=None):
        score=np.asarray(score,float);distance=np.asarray(distance,float)
        sign_margin=self.sign*score-self.protocol['facing_dot_numerical_deadzone']
        band_margin=self.oband-distance
        boundary=distance-self.protocol['query_boundary_tolerance_m']
        passed=(sign_margin>0)&(distance<=self.oband)&(distance>self.protocol['query_boundary_tolerance_m'])
        return {'sign_margin':sign_margin,'band_margin_m':band_margin,'query_boundary_margin_m':boundary,
                'raw_passed':passed,'passed':None if valid is None else passed&np.asarray(valid,bool),
                'solver_margin':np.r_[sign_margin-1e-8,(band_margin-1e-9)/.001,(boundary-1e-9)/.001]}
    def verify(self,value,exact=False):
        x=np.asarray(value['state_scaled'],float);self.check(x.shape==(27,) and np.isfinite(x).all(),'finite complete27 state')
        distance=np.asarray(value['part_distance_m'],float);P=len(self.p);O=len(self.o)
        self.check(distance.shape==(P+O,) and np.isfinite(distance).all() and np.all(distance>=0),'fixed part-distance rows')
        normals=np.asarray(value['part_outward_normal'],float)
        self.check(normals.shape==(P+O,3) and np.isfinite(normals).all(),'fixed raw normal rows')
        self.close(-np.einsum('ij,ij->i',normals[P:],self.on),value['raw_score'],'raw signed score')
        error=self.radius+self.allowance;lower=np.maximum(0.,distance[:P]-error);upper=distance[:P]+error
        margin=np.where(self.near,self.band-upper,lower-self.band)
        proxpass=np.where(self.near,upper<=self.band,lower>self.band)
        prox={'lower_m':lower,'upper_m':upper,'margin_m':margin,'passed':proxpass,
              'solver_margin':(np.where(self.near,margin,(distance[:P]-error)-self.band)-1e-9)/.001,'strict_far_boundary':(~self.near)&(lower==self.band)}
        for k,v in prox.items():self.equal(v,value['proximity'][k],'literal proximity '+k)
        rawori=self.orientation(value['raw_score'],distance[P:])
        for k,v in rawori.items():
            if v is None:self.check(value['raw_orientation'][k] is None,'raw orientation no exact-pass declaration')
            else:self.equal(v,value['raw_orientation'][k],'raw orientation '+k)
        guidance=np.r_[(np.asarray(value['raw_nearest_part_margin_m'])-1e-9)/.001,
                       (np.asarray(value['raw_query_exterior_margin_m'])-1e-9)/.001]
        self.equal(np.r_[prox['solver_margin'],rawori['solver_margin'],guidance],value['surface_margin'],'non-cancelling complete raw row vector')
        errors=np.asarray(value['task_error_vectors_m'],float)
        self.check(errors.shape==(10,3) and np.isfinite(errors).all(),'ten original target error vectors')
        norms=np.sqrt(np.einsum('ij,ij->i',errors[:5],errors[:5]));mean=float(norms.mean());tip=value['tip_status']
        self.equal(norms,tip['norms_m'],'literal tip norms');self.equal(norms<=self.caps,tip['individual_pass'],'original exact caps')
        self.equal(mean,tip['mean_m'],'tip mean');self.equal(mean<=.01,tip['mean_pass'],'original inclusive10mmmean')
        tip_pass=bool(np.all(norms<=self.caps) and mean<=.01);self.equal(tip_pass,tip['passed'],'complete tip feasibility')
        if not exact:return {'tip_pass':tip_pass}
        f=value['exact_orientation_fields'];valid=np.asarray(f['valid'],bool)
        self.check(valid.shape==(O,),'all exact orientation witnesses retained')
        if O:
            self.close(-np.einsum('ij,ij->i',np.asarray(f['outward_normal']),self.on),f['score'],'exact signed normal score')
            masks=[np.asarray(v,bool) for v in f['reason_masks'].values()]
            self.equal(~np.logical_or.reduce(masks),valid,'exact eligibility union')
            self.equal(np.asarray(f['part_index'])!=self.part,f['reason_masks']['different_nearest_part'],'global native nearest part')
            self.check(all(v.shape==(O,) for v in masks),'all invalidation masks preserve row count')
        ori=self.orientation(f['score'],f['distance_m'],valid)
        for k,v in ori.items():self.equal(v,value['exact_orientation'][k],'exact orientation '+k)
        passed=np.r_[proxpass,ori['passed']]
        self.equal(passed,value['literal_surface_pass'],'all literal witness truth')
        self.equal(bool(passed.all()),value['all_witness_constraints_pass'],'whole retained witness truth')
        margin=np.r_[prox['margin_m']/.001,ori['sign_margin'],ori['band_margin_m']/.001,ori['query_boundary_margin_m']/.001]
        self.equal(margin,value['exact_surface_margin_without_optimizer_buffers'],'exact rank margin excludes search buffers')
        rank=[int(not passed.all()),int((~valid).sum()),float(max(0.,np.max(-margin,initial=0.))),int((~passed).sum()),float(np.dot(x,x))]
        self.equal(rank,value['rank'],'frozen literal feasible-first ranking')
        self.equal('PRESENT_CAPPED_EXAMPLES' if O else 'UNAVAILABLE_NO_SAVED_RELIABLE_WITNESS',value['orientation_witness_coverage'],'zero evidence not pass coverage')
        return {'tip_pass':tip_pass,'rank':rank,'all_witness_rows_pass':bool(passed.all()),'failed_proximity_rows':int((~proxpass).sum()),'failed_orientation_rows':int((~ori['passed']).sum()),'invalid_orientation_rows':int((~valid).sum()),'max_exact_original_cap_excess_m':float(np.max(norms-self.caps))}
