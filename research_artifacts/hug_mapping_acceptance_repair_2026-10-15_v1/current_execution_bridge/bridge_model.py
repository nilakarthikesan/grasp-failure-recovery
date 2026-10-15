"""Compiled-model measurements for an independent XML command bridge."""
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from xml_reference import transform, transform_points, hull_vertices, symmetric_vertex_distance


def ident(model,kind,name):
    i=int(mujoco.mj_name2id(model,kind,name))
    if i<0:raise ValueError(f'Missing compiled {kind}: {name}')
    return i


def assign_named(model,data,names,values):
    if len(names)!=len(set(names)):raise ValueError('Duplicate command name')
    rows=[]
    for name,value in zip(names,values,strict=True):
        i=ident(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if model.jnt_type[i]!=mujoco.mjtJoint.mjJNT_HINGE:raise ValueError('Expected scalar revolute command')
        address=int(model.jnt_qposadr[i]);data.qpos[address]=value
        rows.append({'name':name,'joint_id':i,'qpos_address':address,'value_rad':float(value),
                     'compiled_limits_rad':model.jnt_range[i].tolist(),'limited':bool(model.jnt_limited[i])})
    return rows



def joint_metadata_against_source(source,rows,tolerance):
    source_joints={j['name']:j for j in source['joints'] if j['kind']=='revolute'}
    if len(rows)!=len(source_joints) or {r['name'] for r in rows}!=set(source_joints):
        raise ValueError('Incomplete source joint metadata')
    result=[]
    for row in rows:
        wanted=source_joints[row['name']]['limits'];actual=np.asarray(row['compiled_limits_rad'])
        error=float(np.max(np.abs(actual-wanted)))
        result.append({**row,'source_limits_rad':wanted.tolist(),'limit_max_difference_rad':error,
                       'source_compiled_ranges_match':bool(error<=tolerance),
                       'source_compiled_limits_match':bool(row['limited'] and error<=tolerance)})
    return result

def assign_carrier(model,data,position,quaternion):
    joint=ident(model,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')
    if model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('Expected free carrier')
    a=int(model.jnt_qposadr[joint]);data.qpos[a:a+3]=position;data.qpos[a+3:a+7]=quaternion
    if model.nmocap:data.mocap_pos[0]=position;data.mocap_quat[0]=quaternion
    return a


def body_world(model,data,name):
    b=ident(model,mujoco.mjtObj.mjOBJ_BODY,name)
    return transform(data.xpos[b],data.xmat[b].reshape(3,3))


def compare_body_frames(source,frames,model,data,position_tol,rotation_tol):
    rows=[];missing=[]
    for name in source['links']:
        bid=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name))
        if bid<0:missing.append(name);continue
        # URDF's root world is MuJoCo world when that literal name is present.
        T=body_world(model,data,name);wanted=frames[name]
        pe=float(np.linalg.norm(T[:3,3]-wanted[:3,3]));re=float(np.max(np.abs(T[:3,:3]-wanted[:3,:3])))
        rows.append({'link':name,'position_error_m':pe,'rotation_matrix_max_error':re,
                     'passed':bool(pe<=position_tol and re<=rotation_tol),
                     'compiled_world_position_m':T[:3,3].tolist(),'xml_world_position_m':wanted[:3,3].tolist(),
                     'compiled_world_rotation':T[:3,:3].tolist(),'xml_world_rotation':wanted[:3,:3].tolist()})
    # An omitted fixed link can be realized from its nearest surviving ancestor;
    # an omitted revolute link is always a structural failure. Keep this separate
    # from the independently compared surviving compiled bodies.
    bychild={j['child']:j for j in source['joints']};virtual=[]
    for name in missing:
        ancestor=name;relative=np.eye(4);chain=[]
        while int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,ancestor))<0:
            if ancestor not in bychild:raise ValueError(f'Unrealizable omitted link {name}')
            j=bychild[ancestor]
            if j['kind']!='fixed':raise ValueError(f'Compiled model omitted moving link {ancestor}')
            relative=j['origin']@relative;chain.append(j['name']);ancestor=j['parent']
        T=body_world(model,data,ancestor)@relative;wanted=frames[name]
        pe=float(np.linalg.norm(T[:3,3]-wanted[:3,3]));re=float(np.max(np.abs(T[:3,:3]-wanted[:3,:3])))
        virtual.append({'link':name,'surviving_ancestor':ancestor,'fixed_chain':chain,
                        'position_error_m':pe,'rotation_matrix_max_error':re,
                        'passed':bool(pe<=position_tol and re<=rotation_tol),
                        'not_an_independent_compiled_body':True})
    return {'compiled_body_rows':rows,'fixed_virtual_link_rows':virtual,
            'passed':bool(all(r['passed'] for r in rows+virtual))}


def compiled_mesh_support(model,g):
    mid=int(model.geom_dataid[g]);va=int(model.mesh_vertadr[mid]);nv=int(model.mesh_vertnum[mid])
    vertices=model.mesh_vert[va:va+nv].astype(float);a=int(model.mesh_graphadr[mid])
    if a<0:raise ValueError('No compiled mesh support graph; cannot claim collision hull equivalence')
    n=int(model.mesh_graph[a]);ids=model.mesh_graph[a+2+n:a+2+2*n].astype(int)
    if not len(ids) or np.any(ids<0) or np.any(ids>=nv):raise ValueError('Malformed compiled support graph')
    full_extreme=ConvexHull(vertices).vertices
    return vertices,{'mesh_id':mid,'compiled_vertex_count':nv,'graph_support_vertex_count':n,
                         'graph_contains_all_full_hull_extreme_vertex_indices':bool(np.isin(full_extreme,ids).all()),
                         'graph_support_indices':ids.tolist()}


def compare_collisions(source,frames,model,data,source_meshes,tolerances,model_is_hand_only=True):
    kindids={'box':int(mujoco.mjtGeom.mjGEOM_BOX),'cylinder':int(mujoco.mjtGeom.mjGEOM_CYLINDER),
             'sphere':int(mujoco.mjtGeom.mjGEOM_SPHERE),'mesh':int(mujoco.mjtGeom.mjGEOM_MESH)}
    bybody={};expected={};bychild={j['child']:j for j in source['joints']}
    for c in source['collisions']:
        owner=c['link']
        while int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,owner))<0:
            if owner not in bychild or bychild[owner]['kind']!='fixed':raise ValueError('Unresolvable compiled collision owner')
            owner=bychild[owner]['parent']
        expected.setdefault(owner,[]).append(c)
    for g in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g]))
        if name in expected:bybody.setdefault(name,[]).append(g)
    rows=[];countrows=[]
    if model_is_hand_only:
        actual_hand_ids=set(range(model.ngeom))
    else:
        carrier=ident(model,mujoco.mjtObj.mjOBJ_BODY,'forearm')
        def descendant(body):
            while body:
                if body==carrier:return True
                body=int(model.body_parentid[body])
            return False
        actual_hand_ids={g for g in range(model.ngeom) if descendant(int(model.geom_bodyid[g]))}
    expected_owner_ids={g for values in bybody.values() for g in values}
    inventory_matches=actual_hand_ids==expected_owner_ids
    for owner,cs in expected.items():
        gs=bybody.get(owner,[]);countrows.append({'compiled_owner':owner,'source_count':len(cs),'compiled_count':len(gs),'passed':len(cs)==len(gs)})
        if len(cs)!=len(gs):continue
        for c,g in zip(cs,gs,strict=True):
            T=frames[c['link']]@c['origin'];actual=transform(data.geom_xpos[g],data.geom_xmat[g].reshape(3,3))
            row={'source_link':c['link'],'compiled_owner':owner,'source_collision_ordinal':c['ordinal'],'compiled_geom_id':g,
                 'source_kind':c['kind'],'compiled_kind':int(model.geom_type[g]),
                 'contype':int(model.geom_contype[g]),'conaffinity':int(model.geom_conaffinity[g]),
                 'margin_m':float(model.geom_margin[g]),'gap_m':float(model.geom_gap[g]),
                 'correspondence':'source_collision_order_within_named_compiled_owner_including_fixed_body_fusion',
                 'shape_type_matches':bool(model.geom_type[g]==kindids[c['kind']])}
            if not row['shape_type_matches']:row['passed']=False;rows.append(row);continue
            if c['kind']=='mesh':
                points,meta=compiled_mesh_support(model,g)
                actual_world=transform_points(actual,points)
                expected_world=transform_points(T,source_meshes[(c['path'],tuple(c['scale']))])
                error=symmetric_vertex_distance(actual_world,expected_world)
                row.update(mesh=meta,source_mesh_path=c['path'],source_scale=c['scale'].tolist(),
                           symmetric_source_compiled_vertex_set_error_m=error,
                           hull_bound_reason='convex hulls of mutually close finite vertex sets are at most that Hausdorff distance apart, conditional on compiled graph retaining full hull extrema',
                           passed=bool(error<=tolerances['mesh_vertex_set_m'] and meta['graph_contains_all_full_hull_extreme_vertex_indices']))
            else:
                pe=float(np.linalg.norm(actual[:3,3]-T[:3,3]));re=float(np.max(np.abs(actual[:3,:3]-T[:3,:3])))
                se=float(np.max(np.abs(model.geom_size[g]-c['size'])))
                row.update(position_error_m=pe,rotation_matrix_max_error=re,size_error_m=se,
                           passed=bool(pe<=tolerances['current_position_m'] and re<=tolerances['current_rotation_matrix'] and se<=tolerances['primitive_size_m']))
            row['default_contact_contract_matches']=bool(row['contype']==1 and row['conaffinity']==1 and row['margin_m']==0. and row['gap_m']==0.)
            row['passed']=bool(row['passed'] and row['default_contact_contract_matches'])
            rows.append(row)
    current_override_free=bool(model.npair==0 and model.nplugin==0 and model.nflex==0 and model.nexclude==0) if model_is_hand_only else None
    return {'actual_hand_geom_ids':sorted(actual_hand_ids),'expected_owner_geom_ids':sorted(expected_owner_ids),
            'complete_hand_inventory_matches':inventory_matches,
            'current_hand_model_has_no_extra_pair_plugin_flex_or_exclusion':current_override_free,
            'actual_npair':int(model.npair),'actual_nexclude':int(model.nexclude),'actual_nplugin':int(model.nplugin),'actual_nflex':int(model.nflex),
            'count_rows':countrows,'geometry_rows':rows,'source_collision_count':len(source['collisions']),
            'matched_collision_count':len(rows),'passed':bool(inventory_matches and (current_override_free is not False) and len(rows)==len(source['collisions']) and all(r['passed'] for r in rows+countrows)),
            'convex_hull_not_visual_triangle_mesh_equivalence':True}
