"""Pure rigid refresh of immutable compiled collider-local geometry."""
import numpy as np


def refresh_colliders(colliders,model,data,object_position_m,object_rotation):
    """Refresh all dynamic object-frame fields without rebuilding hulls.

    Local triangles/equations/sizes remain immutable. Both the exact local
    query representation and the object-frame triangle cache describe the
    current time-zero FK state. No simulation API is called here.
    """
    pos=np.asarray(object_position_m,float);R=np.asarray(object_rotation,float)
    updated=[]
    for original in colliders:
        g=original.copy();gid=g['id']
        g['R']=R.T@data.geom_xmat[gid].reshape(3,3)
        g['pos']=(data.geom_xpos[gid]-pos)@R
        g['tri_object']=None if g['tri'] is None else g['tri']@g['R'].T+g['pos']
        updated.append(g)
    return updated


def query_anatomical_part(geometry_library,colliders,points):
    """Literal distance/nearest point to individual compatible part surfaces."""
    points=np.asarray(points,float)
    distance=np.full(len(points),np.inf);closest=np.full_like(points,np.nan);normal=np.zeros_like(points)
    for g in colliders:
        signed,cp,n=geometry_library.geom_query(g,points)
        trial=np.abs(signed);update=trial<distance
        distance[update]=trial[update];closest[update]=cp[update];normal[update]=n[update]
    if not np.isfinite(distance).all() or not np.isfinite(closest).all():
        raise ValueError('Anatomical collider distances/closest points unavailable')
    return distance,closest,normal
