"""Practical fit wrapper: five uncapped finger fields; palm diagnostics unchanged.

The original frozen context still validates domain, independent FK, raw bytes,
fixed wrists and frame chains. This wrapper only replaces five named unsigned
visual field columns with uncapped true-triangle queries on the same 856 points.
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np

PROTOTYPE_PATH=Path(__file__).resolve().parents[1]/'uncapped_prototype.py'
spec=importlib.util.spec_from_file_location('practical_uncapped_fit_backend',PROTOTYPE_PATH)
prototype=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prototype;spec.loader.exec_module(prototype)
PARTS=('palm','index','middle','ring','little','thumb','wrist_carrier')
FINGERS=PARTS[1:6]

class UncappedContext:
    def __init__(self,context,geometry):
        self.context=context;self.uncapped=prototype.UncappedFields(context.backend,geometry)
        self.uncapped_source=None;self.uncapped_source_bytes=None
    def __getattr__(self,name):return getattr(self.context,name)
    def initialize_source_fields(self):
        value=self.context.initialize_source_fields().copy()
        value[:,1:6]=self.uncapped.evaluate(self.queries,FINGERS)
        self.uncapped_source=value;self.uncapped_source_bytes=value.tobytes();value.setflags(write=False)
        return value
    def evaluate_fields(self,state,current=None,block=None):
        # Original partial-cache guard checks prove every other branch fixed.
        value=self.context.evaluate_fields(state,current,block)
        parts=(block,) if current is not None and block in FINGERS else FINGERS
        raw=self.uncapped.evaluate(self.queries,parts,
            q24=np.array([state['q'][n] for n in self.names]),palm=state['palm'])
        for j,part in enumerate(parts):value[:,PARTS.index(part)]=raw[:,j]
        self.verify_immutable_source()
        return value
    def verify_immutable_source(self):
        self.context.verify_immutable_source()
        if self.uncapped_source is not None and self.uncapped_source.tobytes()!=self.uncapped_source_bytes:
            raise ValueError('Uncapped source field changed')

def mixed_field_metrics(source,target):
    """Five uncapped finger columns; three explicitly capped palm/wrist fields."""
    s,t=np.asarray(source),np.asarray(target)
    if s.shape!=t.shape or s.ndim!=2 or s.shape[1]!=8 or not np.isfinite(s).all() or not np.isfinite(t).all():
        raise ValueError('Complete finite mixed-scope fields required')
    result={}
    for i,part in enumerate((*PARTS,'declared_palm_signed_union')):
        delta=t[:,i]-s[:,i]
        result[part+'.maximum_abs_difference_m']=float(np.max(np.abs(delta)))
        result[part+'.MSE_m2']=float(np.mean(delta*delta))
    ss,tt=s[:,7],t[:,7]
    crossing=np.where(ss>0,np.maximum(-tt,0),np.where(ss<0,np.maximum(tt,0),np.abs(tt)))
    result['declared_palm_occupancy.crossing_depth_max_m']=float(np.max(crossing))
    result['declared_palm_occupancy.sign_mismatch_count']=int(np.count_nonzero(np.sign(ss)!=np.sign(tt)))
    return result
