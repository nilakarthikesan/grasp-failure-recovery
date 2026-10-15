"""Additional search guidance for existing exact eligibility; no changed predicate."""
import numpy as np
from se3_case import Case as PreviousCase,base
from numba import njit
closest_triangle=base.a.closest_triangle

@njit
def conflicting_mesh_gap(points,tri,normals,mind,chosenq,chosenn,tol,normtol):
    """Positive iff no conflicting triangle lies in the old inclusive tie band.

    The 1m ceiling bounds a positive search margin only. It cannot hide a
    nonpositive conflicting-triangle gap, and is not an acceptance allowance.
    """
    gap=np.ones(len(points))
    for i in range(len(points)):
        for j in range(len(tri)):
            x,y,z,d2=closest_triangle(points[i,0],points[i,1],points[i,2],tri[j,0],tri[j,1],tri[j,2])
            dn=normals[j]-chosenn[i]
            pointgap=np.sqrt((x-chosenq[i,0])**2+(y-chosenq[i,1])**2+(z-chosenq[i,2])**2)
            if np.sqrt(np.dot(dn,dn))>normtol or pointgap>tol:
                gap[i]=min(gap[i],np.sqrt(d2)-(mind[i]+tol))
    return gap

class Case(PreviousCase):
    def raw(self,x):
        raw=super().raw(x)
        hand=base.refresh_colliders(self.original_hand,self.model,self.data,self.opos,self.oR)
        N=len(self.oq); P=len(base.a.PARTS)
        bypart=np.full((N,P),np.inf); closest=np.full((N,P,3),np.nan); normals=np.full_like(closest,np.nan)
        for g in hand:
            signed,c,n=base.a.geom_query(g,self.oq);d=np.abs(signed);part=base.a.PARTS.index(g['part']);update=d<bypart[:,part]
            bypart[update,part]=d[update];closest[update,part]=c[update];normals[update,part]=n[update]
        rows=np.arange(N);parts=bypart.argmin(1);dist=bypart[rows,parts];cp=closest[rows,parts];normal=normals[rows,parts]
        assert np.isfinite(dist).all() and np.isfinite(cp).all()
        p=self.orientation_protocol
        visual_distance,_,_=base.a.nearest_parts(cp,self.visual_tri,np.zeros(len(self.visual_tri),np.int64),1)
        vertices=self.visual_tri.reshape(-1,3)
        within=np.all((cp>=vertices.min(0))&(cp<=vertices.max(0)),axis=1)
        inside=np.zeros(N,bool)
        inside[within],_=base.a.winding_inside(cp[within],self.visual_tri)
        visual_margin=np.where(inside,-visual_distance[:,0],visual_distance[:,0])-p['object_boundary_tolerance_m']
        facet_gap=np.ones(N)
        for g in hand:
            if g['kind']!=base.mujoco.mjtGeom.mjGEOM_MESH:continue
            signed,gcp,gn=base.a.geom_query(g,self.oq);gd=np.abs(signed)
            localq=(self.oq-g['pos'])@g['R'];localcp=(gcp-g['pos'])@g['R'];localn=gn@g['R']
            tn=base.a.tri_info(g['tri'])[0]
            local_gap=conflicting_mesh_gap(localq,g['tri'],tn,gd,localcp,localn,p['tie_distance_tolerance_m'],p['normal_vector_difference_tolerance'])
            # Old exact evaluator checks a mesh's own facet ties only when
            # that mesh is within the global component tie band.
            gap=np.maximum(gd-(dist+p['tie_distance_tolerance_m']),local_gap)
            facet_gap=np.minimum(facet_gap,gap)
        raw.update(raw_closest_visual_exterior_margin_m=visual_margin,raw_mesh_conflicting_facet_gap_m=facet_gap,
                   extended_eligibility_reference='actual_global_nearest_compiled_component_point_normal',
                   original_surface_margin=raw['surface_margin'].copy(),
                   surface_margin=np.r_[raw['surface_margin'],visual_margin/.001,facet_gap/.001])
        return raw
