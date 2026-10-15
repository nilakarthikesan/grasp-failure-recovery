"""Saved construction ablations; no model, FK engine or geometry work at import.

These diagnostic surfaces are not admitted hand models. Stage deltas are
conditional/nonadditive. The neutral LBS surrogate combines omitted pose
correctives and precision effects; the last replay combines serialized XML
placement and source-to-proper-rotation representation, not compiler attribution.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from xml.etree import ElementTree as ET

import numpy as np

HERE=Path(__file__).resolve().parent
STAGES=('native_reference','posed_uncapped','neutral_lbs_uncapped','rigid_neutral_uncapped',
    'rigid_dropped_fixedlabels','rigid_dropped_bonelabels','rigid_hulls_fullprecision',
    'rigid_hulls_obj_decimal','serialized_obj_xml_saved_frames')
PARTS=('palm','index','middle','ring','little','thumb')
GROUPS=((0,),(1,2,3),(4,5,6),(10,11,12),(7,8,9),(13,14,15))
BONE_PART=np.array([0,1,1,1,2,2,2,4,4,4,3,3,3,5,5,5])
CHUNK=256
REFERENCE_LIMIT_M=1e-12
TRIANGLE_TEST_FACTOR=18620
WITNESSES={20:((5932,2,5),(543,3,5)),28:((32591,1,2),(14847,0,5))}
FLAGS=dict(accepted_mapping=False,native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def arrays(path):
    with np.load(path,allow_pickle=False) as value:return {k:value[k].copy() for k in value.files}
def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def transform(vertices,rotation,position):return np.einsum('ij,nj->ni',rotation,vertices,optimize=False)+position
def object_coordinates(world,position,rotation):return np.einsum('ni,ij->nj',world-position,rotation,optimize=False)


def anatomical_labels(weights,faces):
    weights=np.asarray(weights);faces=np.asarray(faces)
    if weights.ndim!=2 or weights.shape[1]!=16 or not np.isfinite(weights).all() or np.any(weights<0):raise ValueError('Finite nonnegative sixteen-bone weights required')
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or np.any(faces<0) or np.any(faces>=len(weights)):raise ValueError('Literal valid triangle indices required')
    grouped=np.column_stack([weights[:,group].sum(axis=1) for group in GROUPS])
    vertex=np.argmax(grouped,axis=1);face_vertex=vertex[faces]
    counts=np.column_stack([(face_vertex==j).sum(axis=1) for j in range(6)])
    means=grouped[faces].mean(axis=1);maximum=counts.max(axis=1)
    labels=np.argmax(np.where(counts==maximum[:,None],means,-np.inf),axis=1).astype(np.int64)
    return vertex,labels


def source_face_subset(native_faces,source_faces):
    """Match by vertex tuple, never assume closed/source face ordering."""
    lookup={}
    for i,face in enumerate(np.asarray(native_faces)):
        key=tuple(sorted(map(int,face)))
        if key in lookup:raise ValueError('Duplicate native triangle support')
        lookup[key]=i
    ids=[]
    for face in np.asarray(source_faces):
        key=tuple(sorted(map(int,face)))
        if key not in lookup:raise ValueError('Source triangle absent from closed native topology')
        ids.append(lookup[key])
    if len(set(ids))!=len(ids):raise ValueError('Duplicate source triangle support')
    return np.asarray(ids,np.int64)


def support_ledger(owner,source):
    owner=np.asarray(owner);source=np.asarray(source)
    keep=np.all(owner[source]==owner[source][:,:1],axis=1)
    referenced=[np.unique(source[keep&(owner[source[:,0]]==i)]) for i in range(16)]
    return keep,[dict(bone=i,assigned_owner_vertices=int((owner==i).sum()),referenced_retained_vertices=len(referenced[i]),
        orphan_owner_vertices=int((owner==i).sum())-len(referenced[i])) for i in range(16)]


def articulated_vertices(neutral,rest,rotations,anchors,weights,owner):
    """Apply SAVED absolute bone transforms; no skeleton/FK reconstruction.

    A common world translation is applied once. Weights are never normalized.
    Float64 arithmetic on the saved neutral geometry is a named surrogate,
    not a recreation of the source float32 pose-corrective computation.
    """
    v=np.asarray(neutral,np.float64);j=np.asarray(rest,np.float64);r=np.asarray(rotations,np.float64);a=np.asarray(anchors,np.float64);w=np.asarray(weights,np.float64)
    if v.ndim!=2 or v.shape[1]!=3 or j.shape!=(16,3) or r.shape!=(16,3,3) or a.shape!=(16,3) or w.shape!=(len(v),16):raise ValueError('Saved neutral/pose array shapes differ')
    if not all(np.isfinite(x).all() for x in (v,j,r,a,w)):raise ValueError('Nonfinite saved articulation operands')
    relative=np.einsum('kij,nkj->nki',r,v[:,None,:]-j[None,:,:],optimize=False)+(a-a[0])[None,:,:]
    lbs=np.einsum('nk,nki->ni',w,relative,optimize=False)+a[0]
    rigid=relative[np.arange(len(v)),owner]+a[0]
    return lbs,rigid


def read_obj(path):
    vertices=[];faces=[]
    for line in Path(path).read_text().splitlines():
        values=line.split()
        if not values or values[0].startswith('#'):continue
        if values[0]=='v' and len(values)==4:vertices.append([float(x) for x in values[1:]])
        elif values[0]=='f' and len(values)==4 and all(x.isdigit() for x in values[1:]):faces.append([int(x)-1 for x in values[1:]])
        else:raise ValueError('Only pinned triangular OBJ vertex/face records supported')
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,np.int64)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(vertices).all() or np.any(faces<0) or np.any(faces>=len(vertices)):raise ValueError('Invalid serialized OBJ geometry')
    return vertices,faces


def xml_mesh_placements(path):
    """Literal precompiler surface placement, not a reconstructed compiler frame."""
    root=ET.parse(path).getroot();result={}
    meshes={node.get('name'):node for node in root.findall('./asset/mesh')}
    for body in root.findall('.//body'):
        name=body.get('name','')
        if not name.startswith('mano_body_'):continue
        bone=int(name.rsplit('_',1)[1]);geoms=body.findall('./geom')
        if len(geoms)!=1 or geoms[0].get('type')!='mesh':raise ValueError('Exactly one pinned mesh geometry per body required')
        geom=geoms[0];mesh=meshes[geom.get('mesh')]
        if any(geom.get(k) is not None for k in ('quat','euler','axisangle','xyaxes','zaxis','fromto')):raise ValueError('Unexpected nonidentity pinned XML geom orientation')
        if any(mesh.get(k) is not None for k in ('refpos','refquat')):raise ValueError('Unexpected mesh reference transform')
        position=np.fromstring(geom.get('pos','0 0 0'),sep=' ');scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        if position.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(position).all() or not np.array_equal(scale,[1.,1.,1.]):raise ValueError('Literal unit mesh scale/finite placement required')
        result[bone]=dict(position=position,scale=scale,path=(Path(path).parent/mesh.get('file')).resolve())
    if set(result)!=set(range(16)):raise ValueError('Sixteen literal XML mesh placements required')
    return result


def reference_control(reference,native,radius,rules,saved_native_fields):
    error=float(np.max(np.abs(reference-native)))
    expected,_=rules.field_status(native,native,radius);actual,_=rules.field_status(native,reference,radius)
    exact=all(np.array_equal(actual[k],expected[k]) for k in expected)
    exact=exact and all(np.array_equal(expected[k],saved_native_fields[k]) for k in ('native_near','native_far','native_ambiguous','required'))
    receipt=dict(max_distance_error_m=error,exact_masks=bool(exact),passed=bool(error<=REFERENCE_LIMIT_M and exact),
        numerical_replay_limit_m=REFERENCE_LIMIT_M,acceptance_allowance_changed=False,
        scope='evaluator/reference consistency only; not hand or physics fidelity')
    return receipt


def _case_source(p,index):
    native=read(p['native_protocol']);v3=read(native['v3_protocol']);v1=read(v3['v1_protocol'])
    plan=next(c for c in v1['cases'] if c['source_inventory_index']==index)
    if len(plan['grids'])!=1:raise ValueError('Exactly the original fixed query grid required')
    return native,plan['grids'][0]


def source_inputs(p):
    """Path reads only; no model, array transform, geometry or field query."""
    paths={Path(__file__),HERE/'test_ablation_case.py',HERE/'README.md',Path(p['native_protocol']),
        Path(p['builder_report']),Path(p['mano_asset_path']),Path(p['geometry_base']),Path(p['field_rules'])}
    builder=read(p['builder_report']);paths.update(Path(row['path']) for row in builder['artifacts'])
    native_run=Path(p['native_run']);paths.update((native_run/'report.json',native_run/'independent_readback_v2.json'))
    for index in (20,28):
        result=native_run/str(index)/'result.json';paths.add(result);paths.update(Path(row['path']) for row in read(result)['artifacts'])
        native,grid=_case_source(p,index);paths.update((Path(native['v3_protocol']),Path(grid['original_cache']),Path(grid['native_cache'])))
    return sorted(path.resolve() for path in paths)


class AblationCase:
    def __init__(self,index,p,raw_case_dir,guard_reinstall=None,budget_guard=None):
        self.artifact_rows=[];self.source_receipt=dict(index=index,diagnostic_only=True,**FLAGS)
        self.counts=dict(full_field_scans=0,triangle_point_tests=0,model_calls=0,FK_engine_calls=0,physics_steps=0,HUG_generator_calls=0,retarget_calls=0,predictor_fits=0)
        if index not in (20,28):raise ValueError('Only the two frozen development cases')
        self.index=index;self.p=p;self.raw=Path(raw_case_dir).resolve();self.raw.mkdir(parents=True,exist_ok=True)
        if any(self.raw.iterdir()):raise FileExistsError('Never overwrite a raw ablation case')
        self.guard=budget_guard or (lambda kind,value:None)
        self.kernel=load('saved_ablation_distance_kernel',p['geometry_base'])
        if guard_reinstall:guard_reinstall()
        self.kernel.load_case=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No model/case construction in saved ablation'))
        import numba
        numba.set_num_threads(2)
        self.rules=load('saved_ablation_frozen_field_rules',p['field_rules'])
        if guard_reinstall:guard_reinstall()
        native,grid=_case_source(p,index);self.grid=grid
        self.result=read(Path(p['native_run'])/str(index)/'result.json');self.decoded=arrays(Path(p['native_run'])/str(index)/'decoded_native_reference.npz')
        self.actual=arrays(Path(p['native_run'])/str(index)/'mesh_full_field.npz')
        self.kinematic=arrays(Path(p['native_run'])/str(index)/'mesh_kinematic_readback.npz')
        builder=read(p['builder_report']);self.generated_root=Path(builder['generated_root'])
        self.producer=arrays(self.generated_root/'neutral_producer_geometry.npz')
        if (not np.array_equal(self.producer['actual_shape_coefficients'].reshape(1,10),self.decoded['shape'].reshape(1,10))
                or not np.array_equal(self.producer['body_joints'],self.decoded['neutral_anchors'])):
            raise ValueError('Saved producer and decoded neutral shape/anchor identities differ')
        # Frozen target labels used original licensed float64 weights, not the
        # producer's float32 tensor. Read only that data key; no MANO layer.
        with Path(p['mano_asset_path']).open('rb') as f:asset=pickle.load(f,encoding='latin1')
        self.target_weights=np.array(asset['weights'],copy=True)
        if self.target_weights.dtype!=np.float64 or self.target_weights.shape!=(778,16):raise ValueError('Original float64 target-label weights required')
        self.query=self.actual['query_object_m'];self.radius=self.actual['cell_radius_m'];self.native_distance=self.actual['native_distance_by_part_m']
        old=arrays(grid['native_cache'])
        if len(self.query)!=grid['query_count'] or not np.array_equal(self.query,old['query_object_m']) or not np.array_equal(self.native_distance,old['native_distance_by_part_m'][:,:6]):raise ValueError('Frozen native/query domain changed')
        original=arrays(grid['original_cache'])
        if not np.array_equal(self.query,original['query_object_m']) or not np.array_equal(self.radius,original['cell_radius_m']):raise ValueError('Frozen query radii changed')
        ref=self.result['source_receipt'];self.object_position=np.asarray(ref['object_reference_world_position_m']);self.object_rotation=np.asarray(ref['object_reference_world_rotation'])
        self.max_tests=len(self.query)*TRIANGLE_TEST_FACTOR
        self.source_receipt.update(query_count=len(self.query),required_relationships=grid['required_relationships'],
            target_weights_dtype=str(self.target_weights.dtype),target_weights_sha256=array_hash(self.target_weights),
            fixed_witnesses=[dict(cell=c,part=PARTS[j],part_index=j,band_mm=b) for c,j,b in WITNESSES[index]])

    def _save(self,name,values,label):
        path=self.raw/name
        with path.open('xb') as f:np.savez_compressed(f,**values)
        row=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,label=label);self.artifact_rows.append(row);return row

    def _geometry(self,world,faces,labels,face_ids,owners,vertex_ids=None):
        faces=np.asarray(faces,np.int64);world=np.asarray(world)
        if vertex_ids is None:vertex_ids=np.arange(len(world))
        return dict(triangles_object_m=object_coordinates(world,self.object_position,self.object_rotation)[faces],
            anatomical_part_index=np.asarray(labels,np.int64),native_face_id=np.asarray(face_ids,np.int64),
            bone_owner=np.asarray(owners,np.int64),source_vertex_ids=np.asarray(vertex_ids)[faces])

    def geometries(self):
        d=self.decoded;producer=self.producer
        full=np.asarray(d['original_native_faces'],np.int64);source=np.asarray(producer['original_faces'],np.int64)
        if full.shape!=(1552,3) or source.shape!=(1538,3):raise ValueError('Frozen closed/source topology sizes changed')
        source_ids=source_face_subset(full,source)
        vertex_labels,labels=anatomical_labels(self.target_weights,full)
        weights=producer['skinning_weights'];owner=np.argmax(weights,axis=1);ownerpart=BONE_PART[owner]
        _,producer_labels=anatomical_labels(weights,full)
        keep,bone_ledger=support_ledger(owner,source)
        self.membership_receipt=dict(closed_faces=1552,source_faces=1538,removed_cap_faces=1552-len(source_ids),
            dropped_crossbone_faces=int((~keep).sum()),retained_faces=int(keep.sum()),
            float64_vs_float32_face_label_disagreements=int((producer_labels!=labels).sum()),
            grouped_label_vs_ownerpart_vertex_disagreements=int((vertex_labels!=ownerpart).sum()),
            by_bone=bone_ledger)
        self._save('membership_provenance.npz',dict(native_vertex_labels=vertex_labels,native_face_labels=labels,
            producer_float32_face_labels=producer_labels,bone_owner=owner,bone_owner_anatomical_part=ownerpart,
            source_face_to_native_face_id=source_ids,retained_source_face_mask=keep),'frozen_membership_and_support')
        posed=d['native_vertices_world_m'];fullowners=np.where(np.all(owner[full]==owner[full][:,:1],axis=1),owner[full[:,0]],-1)
        yield STAGES[0],self._geometry(posed,full,labels,np.arange(len(full)),fullowners)
        yield STAGES[1],self._geometry(posed,source,labels[source_ids],source_ids,fullowners[source_ids])
        neutral=producer['neutral_vertices'].reshape(778,3);rest=producer['body_joints']
        lbs,rigid=articulated_vertices(neutral,rest,d['mano_rotations_world'],d['mano_anchors_world_m'],weights,owner)
        yield STAGES[2],self._geometry(lbs,source,labels[source_ids],source_ids,fullowners[source_ids])
        yield STAGES[3],self._geometry(rigid,source,labels[source_ids],source_ids,fullowners[source_ids])
        kept=source[keep];ids=source_ids[keep];kept_owner=owner[kept[:,0]]
        yield STAGES[4],self._geometry(rigid,kept,labels[ids],ids,kept_owner)
        yield STAGES[5],self._geometry(rigid,kept,BONE_PART[kept_owner],ids,kept_owner)
        placements=xml_mesh_placements(self.generated_root/'assets/literal_native/mano_rhand/mesh_hand.xml')
        hulls=[];obj_hulls=[];replay_hulls=[]
        dtype_receipts=[]
        for i in range(16):
            global_ids=np.flatnonzero(owner==i);local=producer[f'bone_{i}_v_local'];faces=producer[f'bone_{i}_hull_faces'].astype(np.int64)
            expected=neutral[global_ids].astype(local.dtype)-producer['body_centers'][i].astype(local.dtype)
            if not np.array_equal(local,expected):raise ValueError('Producer owner-cloud membership/centering changed')
            obj,of=read_obj(placements[i]['path'])
            serialized=np.array([[float(f'{x:.6f}') for x in row] for row in local])
            if not np.array_equal(of,faces) or not np.array_equal(obj,serialized):raise ValueError('OBJ not exact pinned serialization of saved hull')
            dtype_receipts.append(dict(bone=i,producer_local_dtype=local.dtype.str,producer_local_sha256=array_hash(local),hull_faces_sha256=array_hash(faces)))
            for values,collection in ((local,hulls),(obj,obj_hulls)):
                world=transform(values+producer['body_centers'][i]-rest[i],d['mano_rotations_world'][i],d['mano_anchors_world_m'][i])
                collection.append(self._geometry(world,faces,np.full(len(faces),BONE_PART[i]),np.full(len(faces),-1),np.full(len(faces),i),global_ids))
            world=transform(obj*placements[i]['scale']+placements[i]['position'],self.kinematic['compiled_world_body_rotations'][i],self.kinematic['compiled_body_origins_world_m'][i])
            replay_hulls.append(self._geometry(world,faces,np.full(len(faces),BONE_PART[i]),np.full(len(faces),-1),np.full(len(faces),i),global_ids))
        self.producer_precision=dtype_receipts
        for stage,geoms in zip(STAGES[6:],(hulls,obj_hulls,replay_hulls)):
            yield stage,{key:np.concatenate([g[key] for g in geoms]) for key in geoms[0]}

    def _scan(self,stage,geometry):
        if self.counts['full_field_scans']>=9 or stage!=STAGES[self.counts['full_field_scans']]:raise RuntimeError('Fixed ordered nine-scan quota')
        tri=geometry['triangles_object_m'];labels=geometry['anatomical_part_index']
        cap=1552 if self.counts['full_field_scans']<4 else 1538 if self.counts['full_field_scans']<6 else 3112
        if len(tri)>cap or tri.ndim!=3 or tri.shape[1:]!=(3,3) or not np.isfinite(tri).all() or set(labels)!=set(range(6)):raise ValueError('Complete finite fixed stage geometry required')
        area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        if np.any(area2<=1e-15):raise ValueError('Degenerate diagnostic triangle; no distance inference')
        geometry_binding=self._save(stage+'_geometry.npz',geometry,stage+'_prequery_geometry')
        inventory={part:int((labels==j).sum()) for j,part in enumerate(PARTS)}
        # All operands/membership and fixed witness IDs are sealed before query.
        self.guard('full_field_scan',stage);self.counts['full_field_scans']+=1
        distances=np.empty((len(self.query),6))
        for start in range(0,len(self.query),CHUNK):
            points=self.query[start:start+CHUNK];cost=len(points)*len(tri)
            if self.counts['triangle_point_tests']+cost>self.max_tests:raise RuntimeError('Triangle-point test quota exhausted before query')
            self.guard('triangle_point_tests',cost);self.counts['triangle_point_tests']+=cost
            distances[start:start+len(points)]=self.kernel.nearest_parts(points,tri,labels,6)[0]
        if not np.isfinite(distances).all():raise ValueError('Nonfinite diagnostic field; no inferred pass')
        fields,summary=self.rules.field_status(self.native_distance,distances,self.radius)
        if int(fields['required'].sum())!=self.grid['required_relationships']:raise ValueError('Fixed native relationship denominator changed')
        binding=self._save(stage+'_field.npz',dict(query_object_m=self.query,cell_radius_m=self.radius,
            native_distance_by_part_m=self.native_distance,mapped_distance_by_part_m=distances,**fields),stage+'_complete_fixed_field')
        witness=[]
        for cell,j,band in WITNESSES[self.index]:
            k=(1,2,5).index(band)
            witness.append(dict(cell=cell,part=PARTS[j],band_mm=band,query_object_m=self.query[cell].tolist(),
                native_distance_m=float(self.native_distance[cell,j]),mapped_distance_m=float(distances[cell,j]),
                radius_m=float(self.radius[cell]),classification={key:bool(value[cell,j,k]) for key,value in fields.items()}))
        return dict(stage_id=stage,geometry_artifact=geometry_binding,field_artifact=binding,triangle_count=len(tri),
            face_inventory=inventory,summary=summary,fixed_witnesses=witness,**FLAGS),distances

    def run(self):
        stages=[];control=None;last=None
        for stage,geometry in self.geometries():
            row,distance=self._scan(stage,geometry);stages.append(row);last=distance
            if stage==STAGES[0]:
                control=reference_control(distance,self.native_distance,self.radius,self.rules,self.actual)
                self.source_receipt['reference_control']=control
                if not control['passed']:raise ValueError('Native reference replay failed; stop before causal stages')
        actual=self.actual['mapped_distance_by_part_m']
        residual=dict(max_abs_distance_difference_m=float(np.max(np.abs(last-actual))),
            exact_same_part_field_masks=all(np.array_equal(self.rules.field_status(self.native_distance,last,self.radius)[0][k],self.actual[k]) for k in ('preserved','definite_lost','definite_added','unresolved_required')),
            actual_compiled_mesh_field_status='SAVED_ONLY_NO_NEW_SCAN',
            interpretation='Combined literal serialization/body-frame replay versus saved compiled envelope; compiler-specific effects not isolated')
        return dict(schema_version='manosim_saved_geometry_construction_ablation_case_v1',index=self.index,
            reference_control=control,stages=stages,membership=self.membership_receipt,producer_precision=self.producer_precision,
            source_receipt=dict(self.source_receipt),
            saved_compiled_mesh_residual=residual,counts=dict(self.counts),physical_outcome=None,
            stage_display_names={STAGES[6]:'saved pre-OBJ producer precision; dtype/hash retained'},
            limitations=['Diagnosis only; no candidate admitted','Stage effects conditional and nonadditive',
                'Neutral LBS combines omitted pose correctives and source arithmetic precision',
                'Owner-cloud hull combines convexification and reintroduction of orphan owner vertices',
                'Literal XML plus saved compiled body frames combines XML placement and source-to-proper-rotation representation',
                'Compiled mesh triangles/geom frames were not saved; compiler-specific attribution unavailable',
                'Full orientation and continuous equivalence not proved'],**FLAGS)
