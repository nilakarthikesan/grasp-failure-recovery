from pathlib import Path
out=Path(__file__).parent;old=Path('/private/tmp/hug_guarded_geometry_2026-10-14_v1')
s=(old/'run_guarded_geometry.py').read_text()
s=s.replace('from feasible_surface_refinement import FeasibleRefinementConfig,solve_feasible_refinement','from free_palm_solver import FreePalmConfig,solve_free_palm\nfrom free_palm_kinematics import decode_state,urdf_tip_reaches,state_bounds,initial_seeds,virtual_tip_jacobian,verify_jacobian')
s=s.replace('from grasp_failure_prediction.evaluation.pose_validation import _link_positions,ShadowPoseValidator,_body_id','from grasp_failure_prediction.evaluation.pose_validation import _link_positions,ShadowPoseValidator,_body_id,_fixed_tip_offsets,TASK_LINK_NAMES')
s=s.replace("p['method_version']==3 and p['method_stage']=='fixed_direction_guarded_bounded_pattern_v1'","p['method_version']==1 and p['method_stage']=='free_palm_normalized_pattern_v1'")
s=s.replace("p['solver_configuration']['maximum_pattern_sweeps']==2","p['solver_configuration']['maximum_pattern_sweeps']==4").replace("p['solver_configuration']['maximum_full_geometry_evaluations']==64","p['solver_configuration']['maximum_full_geometry_evaluations']==128")
s=s.replace('FeasibleRefinementConfig(**','FreePalmConfig(**')
a=s.index("        seed_path=Path(p['seed_bindings']")
b=s.index('        limits=np.asarray',a)
s=s[:a]+'''        names=case['stored_shadow_command']['joint_names'];q0=np.array([meta['recovered_original_targets'][n] for n in names])
        assert len(names)==len(q0)==24
        state0=np.zeros(27)
'''+s[b:]
s=s.replace("        full_callback_count=0;tip_callback_count=0",'''        full_callback_count=0;tip_callback_count=0;analytic_callback_count=0;set_pose_calls=0
        dofs=np.asarray([m.jnt_dofadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names],int)
        tipspec=_fixed_tip_offsets(a.URDF)
        tipbodies=np.asarray([_body_id(m,tipspec[n][0]) for n in TASK_LINK_NAMES[:5]],int)''')
a=s.index('        def set_pose(q):');b=s.index('        def evaluate(q,emit=True):',a)
s=s[:a]+'''        def set_pose(state):
            nonlocal set_pose_calls
            set_pose_calls+=1
            values,shift=decode_state(state,q0)
            d.qpos[:]=nominal;d.qpos[adr]=values;d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
            mujoco.mj_forward(m,d)
            target_palm=T_wrist[:3,3]+shift
            corrected=rootpos.copy() if np.array_equal(state,state0) else rootpos+(target_palm-d.xpos[palm])
            d.qpos[rootadr:rootadr+3]=corrected
            if m.nmocap:d.mocap_pos[0]=corrected;d.mocap_quat[0]=rootquat
            mujoco.mj_forward(m,d)
            assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
            assert np.array_equal(d.xpos[object_body],opos) and np.array_equal(d.xmat[object_body].reshape(3,3),oR)
            assert np.linalg.norm(d.xpos[palm]-target_palm)<1e-12
            assert np.array_equal(d.qpos[rootadr+3:rootadr+7],rootquat)
            return corrected
'''+s[b:]
s=s.replace("'joint':surface_config.joint_weight*(np.asarray(q)-q0)/surface_config.joint_delta_limit_rad/np.sqrt(len(q0))}","'joint':surface_config.joint_weight*(decode_state(q,q0)[0]-q0)/surface_config.joint_delta_limit_rad/np.sqrt(len(q0))}")
s=s.replace("'palm_anchor_position_error_m':float(np.linalg.norm(d.xpos[palm]-T_wrist[:3,3]))}","'palm_anchor_position_error_m':float(np.linalg.norm(d.xpos[palm]-(T_wrist[:3,3]+decode_state(q,q0)[1]))),\n                'palm_displacement_world_m':decode_state(q,q0)[1].tolist()}")
a=s.index("        print(json.dumps({'starting':")
b=s.index('        baseline_holder={}',a)
s=s[:a]+'''        def evaluate_tip_jacobian(state):
            nonlocal analytic_callback_count
            analytic_callback_count+=1;set_pose(state)
            return virtual_tip_jacobian(m,d,palm,_link_positions(m,d,a.URDF)[:5],tipbodies,dofs)
        initial_errors=evaluate_tips(state0)
        original_caps=np.sqrt(np.einsum('ij,ij->i',initial_errors,initial_errors))
        bounds,domain=state_bounds(q0,limits,native_tasks[:5],original_caps,T_wrist[:3,3],urdf_tip_reaches(a.URDF))
        seeds,seed_labels=initial_seeds(bounds)
        assert seeds.shape==(26,27)
        jacobian_gate=verify_jacobian(state0,bounds,evaluate_tips,evaluate_tip_jacobian,
            **p['analytic_jacobian_verification'])
        pre_solver_tip_count=tip_callback_count;pre_solver_analytic_count=analytic_callback_count
        print(json.dumps({'starting':caseid,'source_inventory_index':index,'normal_available_parts':available,
            'normal_bins':len(normal_sites),'normal_source_coverage':normal_coverage['exact_original_shared_valid'],
            'jacobian_maxabs_m_per_state':jacobian_gate['maxabs_difference_m_per_scaled_state']}),flush=True)
'''+s[b:]
s=s.replace('KeyedDirectionEvidence(q0,normal_reference,normal_block_start=len(native_distance)*4,normal_block_count=len(normal_sites))',"KeyedDirectionEvidence(state0,normal_reference,normal_block_start=len(native_distance)*4,normal_block_count=len(normal_sites),trace_directory=OUT/'full_callback_traces'/str(index))")
s=s.replace('np.array_equal(q,q0) and not baseline_holder','np.array_equal(q,state0) and not baseline_holder')
s=s.replace('returned,audit=solve_feasible_refinement(q0,limits,evaluate_for_solver,config=configuration,evaluate_fingertips=evaluate_tips,predeclared_seed_qpos_rad=[seed_q],candidate_guard=evidence.assess)', 'returned_state,audit=solve_free_palm(state0,bounds,evaluate_for_solver,config=configuration,evaluate_fingertips=evaluate_tips,predeclared_seed_state_scaled=seeds,candidate_guard=evidence.assess,evaluate_fingertip_jacobian=evaluate_tip_jacobian)\n        returned,selected_shift=decode_state(returned_state,q0)')
s=s.replace("np.array_equal(terminal_holder['qpos_rad'],returned)","np.array_equal(terminal_holder['qpos_rad'],returned_state)")
s=s.replace('evidence.assess(returned,terminal_holder','evidence.assess(returned_state,terminal_holder')
s=s.replace("'predeclared_seed_after_FK_restore' if audit['accepted_candidates'][-1]['proposal_sequence']==1 else 'pattern_after_FK_restore'","'normalized_seed_or_pattern_after_FK_restore'")
s=s.replace("        audit['predeclared_seed_source_sha256']=p['seed_bindings'][str(index)]['old_cohort_sha256']","        audit['seed_labels']=seed_labels\n        assert audit['counters']['FK']==tip_callback_count-pre_solver_tip_count\n        assert audit['counters']['analytic_Jacobians']==analytic_callback_count-pre_solver_analytic_count")
s=s.replace('preflight_errors=evaluate_tips(returned)','preflight_errors=evaluate_tips(returned_state)')
s=s.replace('expected_world=fixedlinks@Rop.T+T_wrist[:3,3]','expected_world=fixedlinks@Rop.T+T_wrist[:3,3]+selected_shift')
s=s.replace("'original_qpos_rad':q0.tolist(),'raw_candidate_qpos_rad':seed_q.tolist(),'predeclared_seed_qpos_rad':seed_q.tolist(),'seed_source_cohort_sha256':p['seed_bindings'][str(index)]['old_cohort_sha256'],","'original_qpos_rad':q0.tolist(),'original_state_scaled':state0.tolist(),'selected_state_scaled':returned_state.tolist(),\n            'palm_displacement_world_m':selected_shift.tolist(),'palm_displacement_norm_m':float(np.linalg.norm(selected_shift)),\n            'state_coordinates':p['state_coordinates'],'domain':domain,'analytic_jacobian_baseline_check':jacobian_gate,")
s=s.replace("'target_palm_world_position_m':T_wrist[:3,3].tolist()","'target_palm_world_position_m':(T_wrist[:3,3]+selected_shift).tolist()")
s=s.replace("'palm_anchor_position_fixed':True","'palm_anchor_position_fixed':False,'palm_orientation_source_convention_unchanged':True")
s=s.replace("'fallback_original_target_exact':bool(np.array_equal(returned,q0))","'fallback_original_target_exact':bool(np.array_equal(returned_state,state0))")
s=s.replace("'full_callback_count_including_fresh_final':full_callback_count,'tip_callback_count':tip_callback_count,","'full_callback_count_including_fresh_final':full_callback_count,'tip_callback_count':tip_callback_count,\n            'analytic_jacobian_callback_count':analytic_callback_count,'set_pose_calls':set_pose_calls,'mj_forward_calls_in_set_pose':2*set_pose_calls,\n            'pre_solver_tip_callback_count':pre_solver_tip_count,'pre_solver_analytic_jacobian_callback_count':pre_solver_analytic_count,")
s=s.replace('hug_guarded_geometry_surface_candidate_cohort_v1','hug_free_palm_geometry_candidate_cohort_v1')
s=s.replace("'solver_implementation_sha256':a.sha(OUT/'feasible_surface_refinement.py')","'solver_implementation_sha256':a.sha(OUT/'free_palm_solver.py')")
s=s.replace("'method_version':3,'method_stage':'fixed_direction_guarded_bounded_pattern_v1'","'method_version':1,'method_stage':'free_palm_normalized_pattern_v1'")
s=s.replace("'scope':'Versioned direction-guarded seed and bounded pattern search on six saved development grasps; maximum 64 full callbacks per case, exact computed caps, two sweeps and fixed radius policy. Independent unchanged scientific audit required.'","'scope':'Versioned static per-grasp translated-palm and joint search, exact original caps and fixed direction guard, six saved development grasps. Independent unchanged scientific audit required.'")
s=s.replace('Direction-guarded bounded pattern study','Per-grasp translated-palm bounded pattern study')
(out/'run_free_palm.py').write_text(s)

s=(out/'keyed_direction_evidence.py').read_text().replace('import hashlib,json','import hashlib,json,gzip\nfrom pathlib import Path')
s=s.replace('normal_block_count):','normal_block_count,trace_directory=None):')
s=s.replace('self._records={};self._callbacks=[];self._assessments=[];self._guard=None','self._records={};self._callbacks=[];self._assessments=[];self._guard=None\n        self._trace_directory=Path(trace_directory) if trace_directory is not None else None\n        if self._trace_directory is not None:self._trace_directory.mkdir(parents=True,exist_ok=False)')
s=s.replace("        self._callbacks.append({'callback_sequence':sequence", "        callback={'callback_sequence':sequence")
s=s.replace("'complete_callback_value_and_facing_block_verified':True})",'''\'complete_callback_value_and_facing_block_verified':True}
        if self._trace_directory is None:
            self._callbacks.append(callback)
        else:
            raw=(json.dumps(callback,sort_keys=True,separators=(',',':'),allow_nan=False)+'\\n').encode()
            destination=self._trace_directory/f'callback_{sequence:04d}.json.gz'
            compressed=gzip.compress(raw,mtime=0)
            with destination.open('xb') as handle:handle.write(compressed)
            self._callbacks.append({'callback_sequence':sequence,'state_scaled':q.tolist(),
                'normal_diagnostics':_json_copy(normal) if sequence==1 else None,
                'trace_path':str(destination),'trace_compressed_sha256':hashlib.sha256(compressed).hexdigest(),
                'trace_uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'trace_uncompressed_bytes':len(raw),
                'complete_value_saved_losslessly':True,'repeat_exact_q_callback':previous is not None})''')
(out/'keyed_direction_evidence.py').write_text(s)
