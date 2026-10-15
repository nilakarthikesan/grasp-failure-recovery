"""Independent source-XML transform and collision geometry reference.

No project loader, MuJoCo, retargeter or simulator imports. URDF transforms use
explicit Rz(yaw) Ry(pitch) Rx(roll) and local-axis Rodrigues rotations. A stored
floating carrier command replaces the world_joint child's world transform;
it is not multiplied by the fixed world_joint origin a second time.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def xyz(value, default=(0.,0.,0.)):
    v=np.asarray(default,float) if value is None else np.fromstring(value,sep=' ',dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all(): raise ValueError('Finite XYZ required')
    return v


def rpy_matrix(rpy):
    r,p,y=np.asarray(rpy,float);cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz@Ry@Rx


def axis_matrix(axis,angle):
    axis=np.asarray(axis,float);norm=np.linalg.norm(axis)
    if not np.isfinite(norm) or norm<=0:raise ValueError('Finite nonzero axis required')
    a=axis/norm;K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


def transform(position=(0,0,0),rotation=None):
    T=np.eye(4);T[:3,3]=position
    if rotation is not None:T[:3,:3]=rotation
    if not np.isfinite(T).all():raise ValueError('Finite transform required')
    return T


def origin(node):
    return transform(xyz(node.get('xyz')),rpy_matrix(xyz(node.get('rpy')))) if node is not None else np.eye(4)


def quaternion_matrix_wxyz(q):
    q=np.asarray(q,float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)==0:raise ValueError('Finite nonzero quaternion required')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def read_urdf(path):
    path=Path(path).resolve();root=ET.parse(path).getroot()
    links=[n.get('name') for n in root.findall('link')]
    if len(set(links))!=len(links):raise ValueError('Duplicate source link name')
    joints=[];collision=[]
    for j in root.findall('joint'):
        kind=j.get('type')
        if kind not in ('fixed','revolute'):raise ValueError(f'Unsupported source joint {kind}')
        limit=j.find('limit');axis=j.find('axis')
        joints.append({'name':j.get('name'),'kind':kind,'parent':j.find('parent').get('link'),
                       'child':j.find('child').get('link'),'origin':origin(j.find('origin')),
                       'axis':xyz(axis.get('xyz') if axis is not None else None,(1.,0.,0.)),
                       'limits':None if kind=='fixed' else np.array([float(limit.get('lower')),float(limit.get('upper'))])})
    if len({j['name'] for j in joints})!=len(joints):raise ValueError('Duplicate source joint name')
    for link in root.findall('link'):
        for n,c in enumerate(link.findall('collision')):
            geometry=list(c.find('geometry'))
            if len(geometry)!=1:raise ValueError('Exactly one shape per source collision required')
            g=geometry[0];kind=g.tag;record={'link':link.get('name'),'ordinal':n,'kind':kind,'origin':origin(c.find('origin'))}
            if kind=='box':record['size']=xyz(g.get('size'))/2.
            elif kind=='sphere':record['size']=np.array([float(g.get('radius')),0.,0.])
            elif kind=='cylinder':record['size']=np.array([float(g.get('radius')),float(g.get('length'))/2.,0.])
            elif kind=='mesh':
                record.update(path=str((path.parent/g.get('filename')).resolve()),scale=xyz(g.get('scale'),(1.,1.,1.)))
            else:raise ValueError(f'Unsupported source collision {kind}')
            collision.append(record)
    children={j['child'] for j in joints};roots=set(links)-children
    if len(roots)!=1:raise ValueError('One source tree root required')
    return {'path':str(path),'links':links,'joints':joints,'collisions':collision,'root':next(iter(roots))}


def named_values(source,names,values):
    names=list(names);values=np.asarray(values,float)
    expected={j['name'] for j in source['joints'] if j['kind']=='revolute'}
    if len(names)!=len(set(names)) or set(names)!=expected or values.shape!=(len(names),) or not np.isfinite(values).all():
        raise ValueError('Every source revolute joint needs exactly one finite named command')
    return dict(zip(names,values))


def forward_xml(source,names,values,carrier_world=None):
    q=named_values(source,names,values);frames={source['root']:np.eye(4)};pending=list(source['joints'])
    while pending:
        progressed=False
        for j in pending[:]:
            if j['parent'] not in frames:continue
            if carrier_world is not None and j['name']=='world_joint':T=np.asarray(carrier_world,float).copy()
            else:
                T=frames[j['parent']]@j['origin']
                if j['kind']=='revolute':T=T@transform(rotation=axis_matrix(j['axis'],q[j['name']]))
            frames[j['child']]=T;pending.remove(j);progressed=True
        if not progressed:raise ValueError('Disconnected or cyclic source tree')
    if set(frames)!=set(source['links']):raise ValueError('Incomplete source frame realization')
    return frames


def source_limit_rows(source,names,values):
    q=named_values(source,names,values);rows=[]
    for j in source['joints']:
        if j['kind']=='fixed':continue
        lo,hi=j['limits'];v=q[j['name']]
        rows.append({'joint_name':j['name'],'value_rad':float(v),'source_limits_rad':[float(lo),float(hi)],
                     'literal_pass':bool(lo<=v<=hi),'violation_rad':float(max(lo-v,v-hi,0.))})
    return rows


def obj_vertices(path,scale):
    vertices=[]
    for line in Path(path).read_text().splitlines():
        fields=line.split()
        if fields and fields[0]=='v':vertices.append([float(v) for v in fields[1:4]])
    v=np.asarray(vertices,float)*np.asarray(scale,float)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all():raise ValueError('Finite volumetric OBJ mesh required')
    return v


def hull_vertices(vertices):
    v=np.asarray(vertices,float);return v[ConvexHull(v).vertices]


def symmetric_vertex_distance(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    return float(max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max()))


def transform_points(T,points):return np.asarray(points)@T[:3,:3].T+T[:3,3]
