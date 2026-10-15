"""Uncapped fitting-field prototype; no native fitting is executed here.

Uses the existing geometric triangle formula and a complete nearest-referenced-
vertex upper bound. An independent validator must continue using its own code.
"""
import numpy as np
from scipy.spatial import cKDTree

MARGIN_M=1e-11

class UncappedSurface:
    def __init__(self,surface,closest_triangle_pairs):
        self.surface=surface;self.closest_triangle_pairs=closest_triangle_pairs
        self.vertex_tree=cKDTree(surface.vertices[np.unique(surface.faces)],copy_data=True)
        self.pair_count=0

    def query(self,points,chunk=128):
        points=np.asarray(points,dtype=np.float64)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
            raise ValueError('Finite Nx3 queries required')
        outputs=[];surface=self.surface
        for start in range(0,len(points),chunk):
            p=points[start:start+chunk];upper=self.vertex_tree.query(p,workers=1)[0]
            candidates=surface.tree.query_ball_point(p,upper+surface.max_radius+MARGIN_M,
                eps=0.,workers=1,return_sorted=True)
            sizes=np.array([len(row) for row in candidates],dtype=np.int64)
            if np.any(sizes==0):raise ValueError('Uncapped nearest-vertex search lost all triangles')
            rows=np.repeat(np.arange(len(p)),sizes)
            ids=np.concatenate([np.asarray(row,dtype=np.int64) for row in candidates])
            delta=p[rows]-surface.centers[ids]
            keep=np.einsum('ij,ij->i',delta,delta)<=(upper[rows]+surface.radii[ids]+MARGIN_M)**2
            rows=rows[keep];ids=ids[keep];self.pair_count+=len(rows)
            distances,_=self.closest_triangle_pairs(p[rows],surface.triangles[ids])
            best=np.full(len(p),np.inf);np.minimum.at(best,rows,distances)
            if not np.isfinite(best).all() or np.any(best>upper+MARGIN_M):
                raise ValueError('Uncapped minimum does not satisfy referenced-vertex upper witness')
            outputs.append(best)
        return np.concatenate(outputs)

class UncappedFields:
    def __init__(self,backend,geometry):
        self.backend=backend
        self.source=[(part,link,UncappedSurface(mesh,geometry.closest_triangle_pairs))
                     for part,link,mesh in backend.source_surfaces]
        self.target=[(part,link,UncappedSurface(mesh,geometry.closest_triangle_pairs))
                     for part,link,mesh in backend.target_surfaces]

    def evaluate(self,points,parts,*,q24=None,palm=None):
        points=np.asarray(points,dtype=np.float64)
        source=q24 is None
        frames=None if source else self.backend._target_frames(q24,palm)
        result=np.full((len(points),len(parts)),np.inf)
        for part,link,mesh in self.source if source else self.target:
            if part not in parts:continue
            if source:local=points
            else:
                matrix=frames[link]
                local=np.einsum('ni,ij->nj',points-matrix[:3,3],matrix[:3,:3])
            result[:,parts.index(part)]=np.minimum(result[:,parts.index(part)],mesh.query(local))
        if not np.isfinite(result).all():raise ValueError('Requested uncapped part missing')
        return result
