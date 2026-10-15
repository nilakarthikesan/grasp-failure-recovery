def build_diagnostic_case(p, index):
    configuration = ConstrainedRefinementConfig(**p['solver_configuration'])
    surface_config = RefinementConfig(**p['surface_configuration'])
    orientation_p = read(OLD / 'orientation/protocol.json')
    physical = {j.get('name'): (float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))) for j in ET.parse(a.URDF).getroot().findall('joint') if j.find('limit') is not None}
    prior = ROOT / 'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
    inventory = read(prior / 'existing_hug_inventory.json')
    fkmap = {c['native_sha256']: c for c in read(prior / 'static_saved_hug_fk.json')['cases']}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        asset = pickle.loads(a.ASSET.read_bytes(), encoding='latin1')
    weights = np.asarray(asset['weights'])
    grouped = np.column_stack([weights[:, ids].sum(1) for ids in a.GROUPS.values()])
    validator = ShadowPoseValidator(default_dex_urdf_root())
    COUNTS['testing'] = True
    for api in ('mj_step', 'mj_step1', 'mj_step2'):
        try:
            getattr(mujoco, api)(None, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Static integration guard failed')
    COUNTS['testing'] = False
    planned = {row['source_inventory_index']: row for row in p['cases']}
    records = []
    tick = time.perf_counter()
    case = inventory['modern_cases'][index]
    assert case['native']['sha256'] == planned[index]['native_sha256']
    meta, ntri, nlabels, nnorm, original_hand, objects, m, d, initial, objadr, opos, oR = a.load_case(case, grouped, fkmap)
    names = case['stored_shadow_command']['joint_names']
    q0 = np.array([meta['recovered_original_targets'][n] for n in names])
    assert len(names) == len(q0) == 24
    limits = np.asarray([physical[n] for n in names])
    assert np.all(q0 >= limits[:, 0]) and np.all(q0 <= limits[:, 1])
    nominal = d.qpos.copy()
    adr = np.array([m.jnt_qposadr[a.ident(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names], int)
    palm = a.ident(m, mujoco.mjtObj.mjOBJ_BODY, 'palm')
    object_body = a.ident(m, mujoco.mjtObj.mjOBJ_BODY, 'object')
    rootadr = int(m.jnt_qposadr[a.ident(m, mujoco.mjtObj.mjOBJ_JOINT, 'world_joint')])
    rootpos = np.asarray(meta['root_target_world_position_m'])
    rootquat = np.asarray(meta['root_target_world_quaternion_wxyz'])
    source = pickle.loads(Path(case['native']['path']).read_bytes())
    T_wc = np.load(case['frame_files']['T_world_camera.npy']['path'], allow_pickle=False)
    T_wrist = T_wc @ np.asarray(source['T_camera_wrist'], float)
    Rop = T_wrist[:3, :3] @ MANO_TO_OPERATOR_RIGHT.T
    native_world = np.asarray(source['landmarks_3d']) @ T_wc[:3, :3].T + T_wc[:3, 3]
    native_tasks = native_world[np.array([4, 8, 12, 16, 20, 2, 6, 10, 14, 18])]
    baseline_anchors = _link_positions(m, d, a.URDF).copy()
    visual_tri, visual_norm, visual_meta = direction.o.visual_mesh(meta, m, d, opos, oR)
    caseid = case['object_id'] + '_' + Path(case['native']['path']).parent.name
    cache = Path(p['query_cache_root']) / f'{caseid}_density2mm_queries.npz'
    assert a.sha(cache) == p['query_cache_sha256'][str(cache)]
    with np.load(cache, allow_pickle=False) as f:
        cached = {k: f[k].copy() for k in f.files}
    fullpath = Path(p['original_full_distance_root']) / f'{caseid}_density2mm_full_current.npz'
    assert a.sha(fullpath) == p['full_cache_sha256'][str(fullpath)]
    with np.load(fullpath, allow_pickle=False) as f:
        full = {k: f[k].copy() for k in f.files}
    query, area, on = (cached['query_object_m'], cached['area_mm2'], cached['visual_triangle_outward_normal'])
    nd, sd = (cached['native_distance_by_part_m'], full['current_distance_by_part_m'])
    assert np.array_equal(full['query_object_m'], query)
    prepared = {}
    for pi, part in enumerate(a.PARTS[:6]):
        sites, siteweights, sitemeta = select_sites(nd[:, pi], sd[:, pi], area, surface_config)
        subset = ntri[nlabels == pi]
        dist, ncp, nix = a.nearest_parts(query[sites], subset, np.zeros(len(subset), np.int64), 1)
        assert np.max(np.abs(dist[:, 0] - nd[sites, pi])) < 1e-12
        prepared[part] = {'query_indices': sites, 'weights': siteweights, 'selection': sitemeta, 'native_distance_m': nd[sites, pi], 'native_closest_object_m': ncp, 'points': query[sites]}
    normal_reference, normal_coverage = direction.freeze_direction_sites(query, area, on, ntri, nlabels, nnorm, original_hand, visual_tri, orientation_p, nd, cached['native_inside'], cached['native_nearest_part'], full, bins_per_part=p['direction_configuration']['bins_per_part'])
    available = [part for part in a.PARTS[:6] if normal_reference[part]['available']]
    normal_sites = np.concatenate([normal_reference[part]['query_indices'] for part in available]) if available else np.empty(0, int)
    expected_parts = np.concatenate([np.full(len(normal_reference[part]['query_indices']), a.PARTS.index(part), int) for part in available]) if available else np.empty(0, int)
    native_distance = np.concatenate([v['native_distance_m'] for v in prepared.values()])
    native_cp = np.concatenate([v['native_closest_object_m'] for v in prepared.values()])
    quad_weights = np.concatenate([v['weights'] / 6 for v in prepared.values()])
    sqrtw = np.sqrt(quad_weights)
    full_callback_count = 0
    tip_callback_count = 0
    last_extra = None

    def set_pose(q):
        values = np.asarray(q, float)
        assert values.shape == (24,) and np.isfinite(values).all()
        d.qpos[:] = nominal
        d.qpos[adr] = values
        d.qpos[rootadr:rootadr + 3] = rootpos
        d.qpos[rootadr + 3:rootadr + 7] = rootquat
        mujoco.mj_forward(m, d)
        corrected = rootpos.copy() if np.array_equal(values, q0) else rootpos - (d.xpos[palm] - T_wrist[:3, 3])
        d.qpos[rootadr:rootadr + 3] = corrected
        if m.nmocap:
            d.mocap_pos[0] = corrected
            d.mocap_quat[0] = rootquat
        mujoco.mj_forward(m, d)
        assert d.time == 0 and np.array_equal(d.qpos[objadr:objadr + 7], initial)
        assert np.array_equal(d.xpos[object_body], opos) and np.array_equal(d.xmat[object_body].reshape(3, 3), oR)
        assert np.linalg.norm(d.xpos[palm] - T_wrist[:3, 3]) < 1e-12
        return corrected

    def evaluate(q, emit=True):
        nonlocal full_callback_count, last_extra
        full_callback_count += 1
        corrected = set_pose(q)
        hand = refresh_colliders(original_hand, m, d, opos, oR)
        distance = []
        closest = []
        for part, v in prepared.items():
            pp = [g for g in hand if g['part'] == part]
            ds, cp, _ = query_anatomical_part(a, pp, v['points'])
            distance.append(ds)
            closest.append(cp)
        distances = np.concatenate(distance)
        points = np.concatenate(closest)
        normal_fields = None
        normal_blocks = []
        normal_diagnostics = {}
        offset = 0
        if len(normal_sites):
            normal_fields = direction.mapped_direction_fields(query[normal_sites], on[normal_sites], hand, visual_tri, orientation_p, expected_part=expected_parts)
        for part in a.PARTS[:6]:
            ref = normal_reference[part]
            if not ref['available']:
                normal_diagnostics[part] = {'available': False, 'fit_bin_count': 0, 'fixed_part_mass': 0.0, 'reason': ref['reason']}
                continue
            count = len(ref['query_indices'])
            sl = slice(offset, offset + count)
            offset += count
            fields = {k: v[sl] if isinstance(v, np.ndarray) and v.ndim else v for k, v in normal_fields.items()}
            residual = native_direction_residual(ref['native_score'], fields['score'], ref['weights'] * 6, fields['valid']) / np.sqrt(6)
            legacy = direction.direction_residual(ref['native_score'], fields, ref['weights'], p['direction_configuration']['score_weight'], p['direction_configuration']['invalid_penalty'])
            assert np.max(np.abs(residual - legacy)) < 1e-14
            normal_blocks.append(residual)
            normal_diagnostics[part] = {'available': True, 'fit_bin_count': count, 'fixed_part_mass': 1 / 6, 'valid_bin_count': int(fields['valid'].sum()), 'invalid_bin_count': int((~fields['valid']).sum()), 'invalid_reason_bin_counts': {key: int(mask[sl].sum()) for key, mask in normal_fields['reason_masks'].items()}, 'native_score': ref['native_score'].tolist(), 'candidate_score': fields['score'].tolist(), 'candidate_valid': fields['valid'].tolist(), 'native_negative_score_bin_count': int((ref['native_score'] < 0).sum()), 'valid_mapped_minus_native_score': (fields['score'][fields['valid']] - ref['native_score'][fields['valid']]).tolist(), 'residual': residual.tolist(), 'invalid_sites_retained_at_nonzero_penalty': True, 'candidate_nearest_part': fields['part_index'].tolist(), 'candidate_distance_m': fields['distance_m'].tolist(), 'candidate_outward_normal': fields['outward_normal'].tolist(), 'candidate_closest_object_m': fields['closest_object_m'].tolist()}
        anchors = _link_positions(m, d, a.URDF)
        components = {'distance': sqrtw * (distances - native_distance) / surface_config.distance_scale_m, 'closest_vector': surface_config.closest_vector_weight * sqrtw[:, None] * (points - native_cp) / surface_config.distance_scale_m / np.sqrt(3), 'facing': np.concatenate(normal_blocks) if normal_blocks else np.empty(0), 'anchor': surface_config.anchor_weight * (anchors - baseline_anchors) / surface_config.anchor_scale_m / np.sqrt(baseline_anchors.size), 'joint': surface_config.joint_weight * (np.asarray(q) - q0) / surface_config.joint_delta_limit_rad / np.sqrt(len(q0))}
        residuals = np.concatenate([v.ravel() for v in components.values()])
        errors = anchors - native_tasks
        evaluation = ConstrainedEvaluation(residuals, errors[:5])
        cost = configuration.f_scale * np.sum(np.hypot(configuration.f_scale, residuals) - configuration.f_scale)
        last_extra = {'objective': float(cost), 'component_sse': {k: float(np.sum(v * v)) for k, v in components.items()}, 'task_link_error_vectors_m': errors.tolist(), 'task_link_error_norm_m': np.linalg.norm(errors, axis=1).tolist(), 'mean_fingertip_error_m': float(np.linalg.norm(errors[:5], axis=1).mean()), 'mean_middle_error_m': float(np.linalg.norm(errors[5:], axis=1).mean()), 'root_target_position_m': corrected.tolist(), 'root_target_quaternion_wxyz': rootquat.tolist(), 'normal_diagnostics': normal_diagnostics, 'normal_bin_count': len(normal_sites), 'normal_valid_bin_count': sum((v.get('valid_bin_count', 0) for v in normal_diagnostics.values())), 'normal_invalid_bin_count': sum((v.get('invalid_bin_count', 0) for v in normal_diagnostics.values())), 'fixed_quadrature_unsigned_distance_mae_m': float(np.dot(quad_weights, np.abs(distances - native_distance))), 'fixed_quadrature_closest_point_mean_error_m': float(np.dot(quad_weights, np.linalg.norm(points - native_cp, axis=1))), 'palm_anchor_position_error_m': float(np.linalg.norm(d.xpos[palm] - T_wrist[:3, 3]))}
        last_extra['component_soft_l1_cost'] = {k: float(configuration.f_scale * np.sum(np.hypot(configuration.f_scale, v) - configuration.f_scale)) for k, v in components.items()}
        last_extra['normal_fields_for_diagnosis'] = {key: normal_fields[key] for key in ('valid', 'score', 'part_index', 'closest_object_m', 'outward_normal', 'nearest_geom_id', 'tie_count', 'reason_masks')} if normal_fields is not None else {}
        if emit and full_callback_count % p['progress_full_evaluation_interval'] == 0:
            print(json.dumps({'progress_case': caseid, 'full_surface_evaluations': full_callback_count, 'elapsed_seconds': time.perf_counter() - tick, 'last_evaluated_objective': float(cost), 'last_evaluated_mean_tip_mm': last_extra['mean_fingertip_error_m'] * 1000, 'last_evaluated_invalid_direction_bins': last_extra['normal_invalid_bin_count'], 'not_necessarily_solver_major_iterate': True}), flush=True)
        return evaluation

    def evaluate_tips(q):
        nonlocal tip_callback_count
        tip_callback_count += 1
        set_pose(q)
        return _link_positions(m, d, a.URDF)[:5] - native_tasks[:5]
    print(json.dumps({'starting': caseid, 'source_inventory_index': index, 'normal_available_parts': available, 'normal_bins': len(normal_sites), 'normal_source_coverage': normal_coverage['exact_original_shared_valid']}), flush=True)
    initial_evaluation = evaluate(q0, emit=False)
    baseline_fresh = jsonable(last_extra)

    def snapshot(q):
        evaluation = evaluate(q, emit=False)
        result = jsonable(last_extra)
        result['residuals'] = evaluation.residuals.tolist()
        return result
    return {'q0': q0, 'baseline': jsonable(last_extra), 'snapshot': snapshot, 'caseid': caseid, 'native_sha256': case['native']['sha256'], 'input_sha256': meta['input_sha256'], 'joint_names': names, 'original_full_callback_count': full_callback_count, 'counter': lambda: full_callback_count, 'time': lambda: float(d.time), 'object_unchanged': lambda: bool(np.array_equal(d.qpos[objadr:objadr + 7], initial))}
