"""Exact-q authoritative metadata cache for the versioned static driver.

A full callback supplies its own complete value and detached diagnostics.
Guard lookups never read mutable latest-callback state. Repeated exact q bytes
must reproduce complete residuals, fingertip vectors and direction diagnostics;
fresh calls advance the sequence only after this exact consistency check.
"""
from copy import deepcopy
import hashlib,json
import numpy as np
from grasp_failure_prediction.evaluation.constrained_surface_refinement import ConstrainedEvaluation,native_direction_residual
from surface_component_guard import freeze_direction_component_guard


def _sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json_copy(value):
    return json.loads(json.dumps(value,allow_nan=False))


class KeyedDirectionEvidence:
    def __init__(self,q0,reference_by_part,*,normal_block_start,normal_block_count):
        self._q0=self._q(q0)
        self._reference=deepcopy(reference_by_part)
        self._start=int(normal_block_start);self._count=int(normal_block_count)
        if self._start<0 or self._count<1:raise ValueError('nonempty fixed facing block required')
        self._records={};self._callbacks=[];self._assessments=[];self._guard=None

    @staticmethod
    def _q(value):
        result=np.array(value,dtype=np.float64,copy=True)
        if result.shape!=(24,) or not np.isfinite(result).all():raise ValueError('24 finite q values required')
        result.setflags(write=False);return result

    @staticmethod
    def _value(value):
        if not isinstance(value,ConstrainedEvaluation):raise ValueError('ConstrainedEvaluation required')
        return ConstrainedEvaluation(value.residuals,value.fingertip_error_vectors_m)

    def _normal(self,diagnostics,value):
        if not isinstance(diagnostics,dict) or not isinstance(diagnostics.get('normal_diagnostics'),dict):
            raise ValueError('normal diagnostics required from this callback')
        supplied=diagnostics['normal_diagnostics']
        if set(supplied)!=set(self._reference):raise ValueError('direction parts changed')
        facing=[];canonical={}
        for part,ref in self._reference.items():
            entry=supplied[part]
            if entry.get('available') is not ref['available']:raise ValueError('direction availability changed')
            if not ref['available']:
                canonical[part]={'available':False};continue
            native=np.asarray(entry['native_score'],float);score=np.asarray(entry['candidate_score'],float)
            valid=np.asarray(entry['candidate_valid'])
            if not np.array_equal(native,ref['native_score']):raise ValueError('native targets changed')
            if valid.dtype.kind!='b':raise ValueError('boolean validity required')
            residual=native_direction_residual(ref['native_score'],score,ref['weights']*6,valid)/np.sqrt(6)
            if not np.array_equal(residual,np.asarray(entry['residual'],float)):
                raise ValueError('direction diagnostics disagree with inherited residual')
            facing.append(residual)
            canonical[part]={'available':True,'native_score':native.tolist(),'candidate_score':score.tolist(),
                'candidate_valid':valid.tolist(),'weights':np.asarray(ref['weights']).tolist(),'residual':residual.tolist()}
        combined=np.concatenate(facing) if facing else np.empty(0)
        if len(combined)!=self._count or self._start+self._count>value.residuals.size:
            raise ValueError('frozen facing block size changed')
        if not np.array_equal(value.residuals[self._start:self._start+self._count],combined):
            raise ValueError('normal metadata does not match complete callback facing block')
        return canonical

    def record(self,q,value,diagnostics):
        q=self._q(q);value=self._value(value);key=q.tobytes()
        diagnostics=_json_copy(diagnostics)
        normal=self._normal(diagnostics,value)
        if not self._callbacks and key!=self._q0.tobytes():raise ValueError('first full callback must be exact q0')
        previous=self._records.get(key)
        if previous is not None:
            if (not np.array_equal(value.residuals,previous['value'].residuals)
                or not np.array_equal(value.fingertip_error_vectors_m,previous['value'].fingertip_error_vectors_m)
                or normal!=previous['normal'] or diagnostics!=previous['diagnostics']):
                raise ValueError('repeated exact q full callback evidence changed')
        sequence=len(self._callbacks)+1
        record={'q':q,'value':value,'normal':normal,'diagnostics':diagnostics,'sequence':sequence}
        if self._guard is None:self._guard=freeze_direction_component_guard(self._reference,normal)
        self._records[key]=record
        self._callbacks.append({'callback_sequence':sequence,'qpos_rad':q.tolist(),'q_float64_sha256':_sha_array(q),
            'residuals':value.residuals.tolist(),'residuals_float64_sha256':_sha_array(value.residuals),
            'fingertip_error_vectors_m':value.fingertip_error_vectors_m.tolist(),
            'fingertips_float64_sha256':_sha_array(value.fingertip_error_vectors_m),
            'normal_diagnostics':_json_copy(normal),'repeat_exact_q_callback':previous is not None,
            'complete_callback_value_and_facing_block_verified':True})

    def assess(self,q,value):
        q=self._q(q);value=self._value(value);record=self._records.get(q.tobytes())
        if record is None:raise ValueError('no full callback evidence for exact q bytes')
        if (not np.array_equal(value.residuals,record['value'].residuals)
            or not np.array_equal(value.fingertip_error_vectors_m,record['value'].fingertip_error_vectors_m)):
            raise ValueError('guard value differs from exact q full callback')
        result=self._guard.assess(record['normal'])
        result.update(metadata_callback_sequence=record['sequence'],q_float64_sha256=_sha_array(q),
            residuals_float64_sha256=_sha_array(value.residuals),
            fingertips_float64_sha256=_sha_array(value.fingertip_error_vectors_m),
            authoritative_exact_q_metadata=True,complete_callback_value_agreement=True)
        self._assessments.append(_json_copy(result));return _json_copy(result)

    @property
    def callback_count(self):return len(self._callbacks)

    @property
    def all_available_baseline_parts_valid(self):
        if self._guard is None:return False
        original=self._callbacks[0]['normal_diagnostics']
        return all(all(entry['candidate_valid']) for entry in original.values() if entry['available'])

    def audit(self):
        return _json_copy({'schema':'exact_q_keyed_direction_evidence_v1','callback_count':len(self._callbacks),
            'distinct_exact_q_count':len(self._records),'normal_block_start':self._start,'normal_block_count':self._count,
            'callbacks':self._callbacks,'guard_assessments':self._assessments,'guard_reference_is_first_actual_q0_callback':True,
            'fresh_terminal_callback_sequence':self._callbacks[-1]['callback_sequence'],
            'last_guard_callback_sequence':self._assessments[-1]['metadata_callback_sequence'],
            'evidence_cache_eviction':False,'scientific_mapping_acceptance':False})
