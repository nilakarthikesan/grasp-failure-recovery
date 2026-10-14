"""Deterministic triangle cells and conditional unsigned-distance area bounds."""
import numpy as np


def triangle_cells(triangles, density_m):
    """Same face/child order as frozen a.subdivide; covers each source facet."""
    blocks=[]; area=[]; face_ids=[]; depths=[]
    areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    for face_id, (face,a) in enumerate(zip(triangles,areas,strict=True)):
        assert a>0
        level=max(0,int(np.ceil(np.log(a/(density_m*density_m))/np.log(4))))
        children=face[None].copy()
        for _ in range(level):
            x,y,z=children[:,0],children[:,1],children[:,2]
            xy,yz,zx=(x+y)/2,(y+z)/2,(z+x)/2
            children=np.concatenate([np.stack([x,xy,zx],1),np.stack([xy,y,yz],1),np.stack([zx,yz,z],1),np.stack([xy,yz,zx],1)])
        blocks.append(children);area.append(np.full(len(children),a/len(children)*1e6));face_ids.append(np.full(len(children),face_id,np.int64));depths.append(level)
    cells=np.concatenate(blocks); q=cells.mean(1)
    radius=np.linalg.norm(cells-q[:,None,:],axis=2).max(1)
    diameter=np.linalg.norm(cells-np.roll(cells,1,axis=1),axis=2).max(1)
    # For affine distance to a fixed center on a convex triangle the maximum
    # is attained at a vertex; these radii cover every point, not just sites.
    return cells,q,np.concatenate(area),radius,diameter,np.concatenate(face_ids),depths


def masks(distance_m,radius_m,band_m,numerical_allowance_m):
    error=radius_m+numerical_allowance_m
    low=np.maximum(0,distance_m-error);high=distance_m+error
    definite=high<=band_m;possible=low<=band_m
    assert np.all(~definite|possible)
    return definite,possible


def area_interval(mask_low,mask_high,area):
    assert np.all(~mask_low|mask_high)
    return {'lower_mm2':float(area[mask_low].sum()),'upper_mm2':float(area[mask_high].sum()),'uncertain_mm2':float(area[mask_high&~mask_low].sum()),'definite_cell_count':int(mask_low.sum()),'possible_cell_count':int(mask_high.sum())}


def pair_intervals(nd,sd,r,area,band,eps):
    nD,nP=masks(nd,r,band,eps);mD,mP=masks(sd,r,band,eps)
    ni=area_interval(nD,nP,area);mi=area_interval(mD,mP,area)
    shared=area_interval(nD&mD,nP&mP,area)
    union=area_interval(nD|mD,nP|mP,area)
    lost=area_interval(nD&~mP,nP&~mD,area)
    new=area_interval(mD&~nP,mP&~nD,area)
    def ratio(numerator,denominator):
        if denominator['lower_mm2']>0:
            return {'lower':numerator['lower_mm2']/denominator['upper_mm2'],'upper':min(1.,numerator['upper_mm2']/denominator['lower_mm2']),'denominator_definitely_nonzero':True}
        return {'lower':None,'upper':None,'denominator_definitely_nonzero':False,'reason':'denominator_definitely_zero' if denominator['upper_mm2']==0 else 'denominator_may_be_zero'}
    pointn=nd<=band;pointm=sd<=band
    an=area[pointn].sum();am=area[pointm].sum();ai=area[pointn&pointm].sum();au=area[pointn|pointm].sum()
    return {'native_area':ni,'mapped_area':mi,'shared_area':shared,'union_area':union,'lost_native_area':lost,'new_mapped_area':new,'whole_relationship_ambiguous_area_mm2':float(area[(nP&~nD)|(mP&~mD)].sum()),'iou_interval':ratio(shared,union),'native_retention_interval':ratio(shared,ni),'mapped_correspondence_interval':ratio(shared,mi),'centroid_estimates':{'native_area_mm2':float(an),'mapped_area_mm2':float(am),'shared_area_mm2':float(ai),'union_area_mm2':float(au),'lost_native_area_mm2':float(area[pointn&~pointm].sum()),'new_mapped_area_mm2':float(area[pointm&~pointn].sum()),'iou':float(ai/au) if au else None,'native_retention':float(ai/an) if an else None}}
