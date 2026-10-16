"""Independent saved-only native pilot coverage, labels and prefix audit.

Imports stdlib and NumPy only. Does not import producer modules, fit models,
invoke MuJoCo/MANO/FK, or query geometry. Actual execution requires a completed
or hard-stopped frozen pilot report supplied explicitly by the root operator.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import subprocess
import sys
import traceback
import numpy as np

SCHEMA = "manosim_native_pilot_cohort_independent_saved_readback_v3"
DOMAIN = "manosim_right_native_67qpos_51qvel_51ctrl_v1"
CATALOG_SHA = "04507e8834364d1451c5c34134d6ca1819efe9da45df5c093d8fa322f897626f"
FOLDS_SHA = "1d7a7549fd75f25d05a1292e3a9d75ff15f8f60ffe88b9d775e93923cb4887a4"
ROLES = ("train", "validation", "test")
IDENTITIES = ("category_id", "object_id", "geometry_group_id", "observation_group_id", "proposal_group_id")
FLAGS = ("full_skin_parity_claimed", "hardware_calibrated", "scientific_training_allowed", "native_task_qualified")
PRIORITY = ("approach_collision", "no_contact", "failed_acquisition", "drop", "slip_during_hold")
QPOS = ("wrist_tx", "wrist_ty", "wrist_tz") + tuple(f"{j}:quat_{a}" for j in ("wrist_rot",) + tuple(f"mano_joint_{i}" for i in range(1,16)) for a in "wxyz")
QVEL = tuple(f"wrist_t{a}:velocity" for a in "xyz") + tuple(f"{j}:angular_{a}" for j in ("wrist_rot",) + tuple(f"mano_joint_{i}" for i in range(1,16)) for a in "xyz")
CTRL = tuple(f"wrist_motor_t{a}" for a in "xyz") + tuple(f"wrist_motor_r{a}" if i == 0 else f"mano_motor_{i}_{a}" for i in range(16) for a in "xyz")
OBSERVABLES = {"domain", "contract_sha256", "qpos_names", "qvel_names", "ctrl_names", "time_s", "action_interval_start_s", "phase", "phase_start_s", "phase_elapsed_s", "hand_qpos", "hand_qvel", "requested_ctrl", "effective_ctrl"}

def require(condition, message):
    if not bool(condition): raise ValueError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def read(path):
    def pairs(rows):
        value = {}
        for k,v in rows:
            require(k not in value, "Duplicate JSON field: " + k); value[k] = v
        return value
    def reject(value): raise ValueError("Nonfinite JSON constant: " + value)
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject)
    def finite(x):
        if isinstance(x, float): require(np.isfinite(x), "Nonfinite JSON exponent")
        elif isinstance(x, dict):
            for v in x.values(): finite(v)
        elif isinstance(x, list):
            for v in x: finite(v)
    finite(value); return value

def binding(path):
    p = Path(path).resolve(); return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)

def verify_binding(row):
    p = Path(row["path"])
    require(p.is_absolute() and not any(q.is_symlink() for q in (p,*p.parents)), "Aliased/nonabsolute binding")
    require(sha(p) == row["sha256"].removeprefix("sha256:"), "Binding hash changed: " + str(p))
    if "bytes" in row: require(p.stat().st_size == row["bytes"], "Binding length changed")
    return p

def arrays(path):
    with np.load(path, allow_pickle=False) as f: result = {k:f[k].copy() for k in f.files}
    for k,v in result.items():
        require(v.dtype.kind in "biufSU", "Unsupported array dtype: " + k)
        if v.dtype.kind not in "SU": require(np.isfinite(v).all(), "Nonfinite saved operand: " + k)
    return result

def git(root,*args): return subprocess.check_output(["git","-C",str(root),*args])

def closure(root, protocol, p, freeze):
    require(git(root,"rev-parse","HEAD").decode().strip() == freeze, "Exact HEAD hold required")
    for path,digest in p["active_bindings"].items(): require(sha(path) == digest, "Active input changed: " + path)
    for path,digest in p["selected_git_sources"].items():
        require(hashlib.sha256(git(root,"show",freeze+":"+path)).hexdigest() == digest, "Frozen source blob differs: " + path)
    require(git(root,"show",freeze+":"+str(protocol.relative_to(root))) == protocol.read_bytes(), "Protocol Git blob differs")
    return dict(active_bindings=len(p["active_bindings"]), selected_git_sources=len(p["selected_git_sources"]), exact_HEAD=freeze, protocol_blob_verified=True)

def journal_inventory(raw, report):
    rows = report["raw_artifacts"]; require(len(rows) == len({r["path"] for r in rows}), "Duplicate final artifact path")
    indexed = {r["path"]:r for r in rows}
    for row in rows:
        require(Path(row["path"]).is_relative_to(raw), "Artifact escaped raw root"); verify_binding(row)
    actual = {str(x.resolve()) for x in raw.rglob("*") if x.is_file()}
    require(not any(x.is_symlink() for x in raw.rglob("*")) and actual == set(indexed), "Raw inventory is not exhaustive")
    jp = verify_binding(report["creation_time_journal"]); live = {}; events = []
    for line in jp.read_text().splitlines():
        row = json.loads(line); events.append(row)
        if row.get("event") == "artifact_relocation":
            old = row["original_path"]; require(old in live and row["path"] not in live, "Invalid relocation chain")
            original = live.pop(old)
            require(original["sha256"] == row["sha256"] and original["bytes"] == row["bytes"], "Relocation changed creation proof")
            original = {**original,"path":row["path"]}; live[row["path"]] = original
        else:
            require(row["path"] not in live, "Duplicate write-time publication"); live[row["path"]] = row
    require(set(live) == set(indexed), "Creation journal/final inventory differs")
    for path,row in live.items(): require(all(row[k] == indexed[path][k] for k in ("path","sha256","bytes","label")), "Creation-time bytes changed")
    return indexed, events

def plan_coverage(p, catalog):
    objects = {o["object_id"]:o for o in catalog["objects"]}; rows = p["schedule"]
    require(len(objects) == 30 and len(rows) == 135 and len({r["attempt_id"] for r in rows}) == 135, "Exact 30/135 denominator required")
    primary = [r for r in rows if r["cohort"] == "primary"]; matched = [r for r in rows if r["cohort"] == "matched_1ms"]
    require(len(primary) == 120 and len(matched) == 15 and len(primary)+len(matched) == len(rows), "Primary/matched cohort mismatch")
    require({r["object_id"] for r in primary} == set(objects), "Catalog object coverage differs")
    cells = defaultdict(list)
    for oid,obj in objects.items():
        these = [r for r in primary if r["object_id"] == oid]
        require(len(these) == 4 and len({r["observation_id"] for r in these}) == 2, "Each object needs 2 views x 2 seeds")
        require({(r["observation_id"],r["seeds"]["generator"]) for r in these} == {(v,s) for v in {r["observation_id"] for r in these} for s in (42,43)}, "View/seed Cartesian product differs")
        for row in these:
            require((row["category_id"],row["size"],row["replica"]) == (obj["official_geometry_category"],obj["official_size_bin"],obj["cell_replica"]), "Author-provided metadata join differs")
            require(row["dt_s"] == .002 and row["scenario"]["command_noise"] == row["scenario"]["sensor_noise"] == 0, "Primary fixed physics/noise protocol changed")
        cells[(obj["official_geometry_category"],obj["official_size_bin"])].append(oid)
    require(len(cells) == 15 and all(len(v) == 2 for v in cells.values()), "Geometry x size 2-object cells differ")
    byid = {r["attempt_id"]:r for r in primary}
    for r in matched:
        a = byid[r["matched_primary_id"]]
        require(r["dt_s"] == .001 and r["proposal"] == a["proposal"] and r["object_id"] == a["object_id"] and r["replica"] == "A" and r["seeds"]["generator"] == 42, "Matched row identity changed")
        candidates=[v for v in primary if v["object_id"]==r["object_id"] and v["seeds"]["generator"]==42]
        require(a==min(candidates,key=lambda v:p["observations"][v["observation_id"]]["path"]),"Matched observation selection differs")
    require(len({(r["category_id"],r["size"]) for r in matched}) == 15, "Matched per-cell coverage differs")
    return primary, matched, objects, cells

def observed_recording(folder, row, contract_expected=None):
    payload = read(folder/"attempt.json"); meta = payload["attempt"]; c = payload["contract"]; digest = canonical(c)
    require(payload["schema_version"] == "manosim_native_recording_v1" and payload["contract_sha256"] == meta["contract_sha256"] == digest, "Recording contract identity differs")
    require(meta["attempt_id"] == row["attempt_id"] and meta["object_id"] == row["object_id"] and meta["observation_group_id"] == row["observation_id"] and meta["geometry_group_id"] == row["geometry_group_id"] and meta["proposal_group_id"] == row["proposal_group_id"], "Recording identity/group join differs")
    require(meta["source_generator"] == "hug" and meta["domain"] == DOMAIN and c["domain"] == DOMAIN, "Native HUG domain differs")
    hashes=meta["provenance_hashes"]
    for name,value in (("native_pose","sha256:"+row["proposal"]["sha256"]),("capture",row["capture_sha256"]),("scenario",row["scenario_sha256"]),("seeds",canonical(row["seeds"]))):
        require(hashes[name]==value,"Native recording provenance differs: "+name)
    if contract_expected is not None: require(c == contract_expected, "Scheduled recording contract differs")
    require(payload["native_admission"]["admitted"] is False and payload["native_admission"]["contract_sha256"] == digest, "Recording falsely grants admission")
    files = payload["artifacts_sha256"]; require(set(files) <= {"observables.npz","diagnostics.npz"}, "Unexpected recording artifact")
    require({x.name for x in folder.iterdir()} == {"attempt.json",*files}, "Unindexed recording file")
    for name,d in files.items(): require("sha256:"+sha(folder/name) == d, "Recording artifact hash differs")
    a = arrays(folder/"observables.npz") if "observables.npz" in files else None
    if meta["mapping"]["status"] != "ok": require(meta["execution_status"] == "not_executed" and meta["outcome"] is None and a is None, "Mapping refusal contains outcome/observations")
    elif meta["execution_status"] != "completed": require(meta["outcome"] is None, "Runtime error became physical failure")
    else: require(meta["outcome"] is not None and a is not None, "Completed task missing outcome/observations")
    if a is None: require(payload["recording"] is None, "Unobserved sample count fabricated"); return meta,c,None
    require(set(a) == OBSERVABLES, "Feature observable allowlist differs")
    require(str(a["domain"].item()) == DOMAIN and str(a["contract_sha256"].item()) == digest, "Array domain/contract differs")
    for key,names in (("qpos_names",QPOS),("qvel_names",QVEL),("ctrl_names",CTRL)):
        require(a[key].shape == (len(names),) and tuple(a[key].tolist()) == names, "Named native channel order differs: "+key)
    t = a["time_s"]; s = a["action_interval_start_s"]; dt = c["control_interval_s"]; n = len(t)
    require(dt == .01 and t.shape == s.shape == (n,) and n > 0 and s[0] == 0 and np.all(np.diff(t)>0) and np.all(t>s), "Recording clock malformed")
    require(np.max(np.abs(t-s-dt)) <= 1e-9 and (n == 1 or np.max(np.abs(s[1:]-t[:-1])) <= 1e-9), "Recording interval clock differs")
    require(payload["recording"] == dict(samples=n), "Recording sample count differs")
    for key,width in (("hand_qpos",67),("hand_qvel",51),("requested_ctrl",51),("effective_ctrl",51)):
        require(a[key].shape == (n,width), "Native array shape differs: "+key)
    require(np.max(np.abs(np.linalg.norm(a["hand_qpos"][:,3:].reshape(-1,16,4),axis=-1)-1)) <= 1e-8, "Raw quaternion norm differs")
    starts = np.empty(n); current = 0.
    for i in range(n):
        if i and a["phase"][i] != a["phase"][i-1]: current = s[i]
        starts[i] = current
    require(np.array_equal(starts,a["phase_start_s"]) and np.max(np.abs(t-starts-a["phase_elapsed_s"])) <= 1e-9, "Phase clocks differ")
    return meta,c,a

def label_from_saved(st, ft, dt):
    n = int(round(5.5/dt)); t = st["state_time_s"]; f = ft["force_time_s"]
    require(t.shape == (n+1,) and f.shape == (n,) and np.max(np.abs(t-np.arange(n+1)*dt)) <= 1e-9 and np.max(np.abs(f-np.arange(n)*dt)) <= 1e-9, "Label complete physics clocks differ")
    obj=st["object_com_world_m"]; wrist=st["wrist_position_world_m"]; R=st["wrist_rotation_world"]
    require(obj.shape == wrist.shape == (n+1,3) and R.shape == (n+1,3,3), "Label geometry layout differs")
    require(np.max(np.abs(np.swapaxes(R,1,2)@R-np.eye(3))) <= 1e-8 and np.max(np.abs(np.linalg.det(R)-1)) <= 1e-8, "Label wrist rotations improper")
    ho=ft["hand_object_normal_force_N"]; ht=ft["hand_table_normal_force_N"]; ot=ft["object_table_normal_force_N"]
    require(all(x.shape == (n,) and np.all(x>=0) for x in (ho,ht,ot)), "Label force magnitudes/layout differs")
    close,lift,hold = (int(round(s/dt)) for s in (1.,1.5,2.5)); ticks=np.arange(n); z0=float(obj[0,2]); events=[]
    def add(kind,i,clock): events.append(dict(failure_type=kind,physics_tick=int(i),time_s=float(i*dt),clock=clock))
    acq=(ticks>=close)&(ticks<=lift); contacts=ho>.001; supports=ot>.001
    hit=np.flatnonzero(((ticks<close)&contacts)|(ht>.001))
    if hit.size:add("approach_collision",int(hit[0]),"force")
    if not np.any(acq&contacts):add("no_contact",lift,"state_phase_boundary")
    if obj[hold,2]<z0+.10:add("failed_acquisition",hold,"state")
    if supports[hold]:add("failed_acquisition",hold,"force")
    hit=np.flatnonzero((ticks>hold)&supports)
    if hit.size:add("drop",int(hit[0]),"force")
    raised=np.flatnonzero(obj[:,2]>=z0+.10)
    if raised.size:
        hit=np.flatnonzero((np.arange(n+1)>=raised[0])&(obj[:,2]<z0+.08))
        if hit.size:add("drop",int(hit[0]),"state")
    relative=(np.swapaxes(R,1,2)@(obj-wrist)[...,None])[...,0]; shifts=np.linalg.norm(relative[hold:]-relative[hold],axis=1)
    hit=np.flatnonzero(shifts>.02)
    if hit.size:add("slip_during_hold",hold+int(hit[0]),"state")
    events.sort(key=lambda e:(e["physics_tick"],PRIORITY.index(e["failure_type"])))
    outcome=dict(success=not events,failure_type=events[0]["failure_type"] if events else None,label_rule_id="manosim_native_grasp_lift_task_labels_v1",first_failure_time_s=events[0]["time_s"] if events else None)
    return outcome,events,dict(maximum_hold_relative_shift_m=float(shifts.max()),minimum_hold_rise_m=float((obj[hold:,2]-z0).min()))

def feature_prefix(a, outcome, mapping_status, execution_status):
    if mapping_status != "ok": return "mapping_or_setup_refusal", None
    if execution_status != "completed": return "runtime_or_unexecuted", None
    if outcome["failure_type"] == "approach_collision": return "already_failed_approach_collision", None
    if not outcome["success"] and outcome["first_failure_time_s"] is None:return "unknown_failure_time",None
    phases=a["phase"]; times=a["time_s"]; lifts=np.flatnonzero(phases=="lift")
    if not lifts.size:return "no_lift_boundary",None
    first=int(lifts[0]); lift=float(a["action_interval_start_s"][first]); require(abs(lift-1.5)<=1e-9,"Actual scheduled lift boundary differs")
    require(not np.any(np.isin(phases[:first],["hold","score"])) and not np.any(phases[first:]=="close_fingers"),"Invalid phase order")
    cutoff=lift-.01; eligible=np.flatnonzero(times<=cutoff+1e-9)
    if len(eligible)<20 or abs(float(times[eligible[-1]])-cutoff)>1e-9:return "missing_close_history",None
    idx=eligible[-20:]; require(np.all(phases[idx]=="close_fingers") and np.all(times[idx]<lift),"Post-cutoff/nonclosing history")
    if not outcome["success"] and outcome["first_failure_time_s"]<=cutoff+1e-9:return "already_failed_at_or_before_cutoff",None
    pose=a["hand_qpos"][idx].copy(); q=pose[:,3:].reshape(-1,16,4)
    for row in q:
        for ball in row:
            nz=np.flatnonzero(ball);require(len(nz)>0,"Zero quaternion");ball*=(-1 if ball[nz[0]]<0 else 1)
    X=np.concatenate((pose,a["hand_qvel"][idx],a["effective_ctrl"][idx]),axis=1).astype(np.float32)
    require(X.shape==(20,169) and np.isfinite(X).all(),"Feature overflow/shape")
    return None,dict(cutoff_s=cutoff,observed_times_s=times[idx].tolist(),shape=list(X.shape),dtype=str(X.dtype),feature_array_sha256=hashlib.sha256(X.tobytes()).hexdigest(),success=int(outcome["success"]))

def fold_readiness(folds, primary, evaluated):
    byid={r["attempt_id"]:r for r in primary}; result=[]
    for fold in folds:
        assignments=fold["assignments"];require(set(assignments)==set(byid) and set(assignments.values())==set(ROLES),"Fold must assign every primary only")
        for key in ("object_id","geometry_group_id","observation_id","proposal_group_id"):
            groups={role:{byid[k][key] for k,v in assignments.items() if v==role} for role in ROLES}
            require(all(not(groups[a]&groups[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))),"Cross-role group leak: "+key)
        roles={}
        for role in ROLES:
            ids=[k for k,v in assignments.items() if v==role]; rows=[evaluated[k] for k in ids]; usable=[x for x in rows if x.get("prefix") is not None]
            roles[role]=dict(planned_attempts=len(ids),planned_objects=len({byid[k]["object_id"] for k in ids}),status_counts=dict(Counter(x["status"] for x in rows)),physical_success=sum(x.get("success") is True for x in rows),physical_failure=sum(x.get("success") is False for x in rows),null_outcomes=sum(x.get("success") is None for x in rows),eligible=len(usable),eligible_successes=sum(x["success"] is True for x in usable),eligible_future_failures=sum(x["success"] is False for x in usable),exclusions=dict(Counter(x["exclusion"] for x in rows if x.get("exclusion"))),eligible_objects=len({x["object_id"] for x in usable}))
        usable=all(v["eligible_successes"]>0 and v["eligible_future_failures"]>0 for v in roles.values())
        result.append(dict(fold_id=fold["fold_id"],roles=roles,data_status="TWO_CLASS_EACH_ROLE" if usable else "EMPTY_OR_SINGLE_CLASS_ROLE",scientific_training_allowed=False))
    return result

def camera_cast_check(a):
    """Exact stored source64 -> float32 preprocessing, no tolerance or fitting."""
    K=a['source_capture_K64'];expected=a['expected_generator_K32'];saved=a['saved_proposal_K'];T=a['source_capture_cameraWorld64'];nativeT=a['saved_proposal_cameraWorld64']
    require(K.shape==(3,3) and K.dtype==np.float64 and expected.shape==(3,3) and expected.dtype==np.float32 and saved.shape==(3,3) and saved.dtype==np.float32,'Literal camera K dtype/shape differs')
    require(T.shape==nativeT.shape==(4,4) and T.dtype==nativeT.dtype==np.float64,'Literal camera world dtype/shape differs')
    require(expected.tobytes()==K.astype(np.float32).tobytes(),'Stored K32 is not exact source numeric cast')
    passed=bool(saved.tobytes()==expected.tobytes() and nativeT.tobytes()==T.tobytes())
    require(a['capture_camera_and_K32_cast_pass'].shape==() and bool(a['capture_camera_and_K32_cast_pass'])==passed,'Stored camera-cast decision differs')
    return passed

def preserved_v1(root,p):
    prior=p['prior_v1_refusals'];pp=verify_binding(prior['protocol']);rp=verify_binding(prior['report']);mp=verify_binding(prior['mechanical_readback']);cp=verify_binding(prior['cohort_readback'])
    require(sha(rp)=='55ced44bb921f1ae0a76dd46768de3920132ece4a28ae9a75b552d2bbe2fa9c6' and sha(mp)=='d093594729839b18d70a480b0e9125207e6b069d2a9c9f9f462295e1f5eac726' and sha(cp)=='8caeb6dcc973b3488c6bb5097013ffa567c59232e398cdacd261e93dc4b1636b','Exact preserved v1 receipts required')
    old=read(pp);report=read(rp);cohort=read(cp);freeze='959db6c119b67dbf66a6e2ad67f0da36b36d88eb'
    require(report['freeze_commit']==freeze and report['status']=='COMPLETED' and report['actual_counts']['steps']==report['actual_counts']['contact_force_reads']==0 and all(x['status']=='MAPPING_REFUSAL' and x['outcome'] is None for x in report['attempt_schedule']),'Original v1 refusal truth differs')
    require(cohort['schema_version']=='manosim_native_pilot_cohort_independent_saved_readback_v1' and cohort['primary']['eligible']==0 and cohort['scientific_training_allowed'] is False,'Original v1 readiness differs')
    require(git(root,'show',freeze+':'+str(pp.relative_to(root)))==pp.read_bytes(),'Original v1 protocol Git blob differs')
    for rel,digest in old['selected_git_sources'].items():require(hashlib.sha256(git(root,'show',freeze+':'+rel)).hexdigest()==digest,'Original v1 source blob differs')
    for row in [report['worker_result'],report['creation_time_journal'],*report['raw_artifacts']]:verify_binding(row)
    return dict(protocol=binding(pp),report=binding(rp),cohort_readback=binding(cp),mechanical_readback=binding(mp),archived_selected_source_blobs=len(old['selected_git_sources']),original_null_refusals=135,original_steps=0,historical_runtime_live_reverified=False)

def verify(root, protocol, freeze, report_sha):
    p=read(protocol);require(p['schema_version']=='manosim_native_object_pilot_v3_protocol','Exact v3 protocol required');require(p['collection_scope']=='CONDITIONAL AUTHORED HUG ASSET SIMULATION; native capsule approximation; RGBD physical scene correspondence NOT calibrated; no full-skin or hardware transfer claim','Explicit conditional scene scope required');before=closure(root,protocol,p,freeze);prior=preserved_v1(root,p);rp=protocol.parent/"run_v1/report.json";require(sha(rp)==report_sha,"Exact parent report hash required");r=read(rp)
    require(r['schema_version']=='manosim_native_object_pilot_v3_report','Exact v3 report required')
    require(r["protocol_sha256"]==sha(protocol) and r["freeze_commit"]==freeze and r["status"] in ("COMPLETED","HARD_STOP"),"Report transaction identity differs")
    require(r["training_gate"]=="CLOSED_PENDING_INDEPENDENT_NATIVE_TASK_QUALIFICATION" and r["physical_outcome"] is None and all(r[k] is False for k in FLAGS),"Unexpected training/admission")
    raw=Path(p["raw_root"]);catpath=verify_binding(p["catalog_binding"]);foldpath=verify_binding(p["catalog_folds_binding"])
    require(sha(catpath)==CATALOG_SHA and sha(foldpath)==FOLDS_SHA,"Fixed official catalog/folds changed");cat=read(catpath);foldspec=read(foldpath)
    require(p["catalog_folds_spec"]==foldspec,"Embedded source fold specification differs")
    primary,matched,objects,cells=plan_coverage(p,cat);artifacts,events=journal_inventory(raw,r)
    w=read(verify_binding(r["worker_result"]));require(w["attempt_schedule"]==r["attempt_schedule"] and w["artifacts"]==r["raw_artifacts"] and w["actual_counts"]==r["actual_counts"],"Worker/parent copies differ")
    ledger=r["attempt_schedule"];require(len(ledger)==135 and [x["attempt_id"] for x in ledger]==[x["attempt_id"] for x in p["schedule"]],"Allcase ordered ledger differs")
    ledger={x["attempt_id"]:x for x in ledger};evaluated={};prefix_hash=hashlib.sha256();complete_labels=0;static_counts=Counter();camera_rows=[]
    for row in p["schedule"]:
        aid=row["attempt_id"];status=ledger[aid]["status"];entry=dict(attempt_id=aid,object_id=row["object_id"],category_id=row["category_id"],size=row["size"],cohort=row["cohort"],status=status,reason_code=ledger[aid].get('reason_code'),success=None,exclusion="runtime_or_unexecuted",prefix=None)
        folder=raw/aid;record=folder/"recording/attempt.json"
        gate=folder/'source_chart_gate.npz'
        if gate.is_file():
            gate_arrays=arrays(gate);camera_pass=camera_cast_check(gate_arrays)
            require(ledger[aid]['static_checks']['capture_camera_and_K32_cast']==camera_pass,'Ledger camera-cast decision differs')
            native=arrays(verify_binding(row['proposal']));require(native['K'].dtype==gate_arrays['saved_proposal_K'].dtype and native['K'].tobytes()==gate_arrays['saved_proposal_K'].tobytes() and native['cameraWorld64'].tobytes()==gate_arrays['saved_proposal_cameraWorld64'].tobytes(),'Camera gate not bound to actual saved proposal')
            camera_rows.append(dict(attempt_id=aid,exact_K32_cast_and_camera_passed=camera_pass,source_K64_dtype=str(gate_arrays['source_capture_K64'].dtype),saved_K_dtype=str(native['K'].dtype)))
        for key,value in ledger[aid].get("static_checks",{}).items():
            require(type(value) is bool,"Static check not a boolean");static_counts[key+":"+("PASS" if value else "FAIL")]+=1
        if record.is_file():
            meta,c,a=observed_recording(record.parent,row);out=meta["outcome"]
            if status=="COMPLETED":
                require(meta["execution_status"]=="completed" and out==ledger[aid]["outcome"],"Completed recording/ledger outcome differs")
                st=arrays(folder/"physics_state.npz");ft=arrays(folder/"physics_forces.npz");expected,fe,metrics=label_from_saved(st,ft,row["dt_s"]);lr=read(folder/"label_report.json")
                require(expected==out==lr["outcome"] and lr["status"]=="SCORED" and lr["scientific_training_admitted"] is False,"Independent physical label differs")
                require([{k:e[k] for k in ("failure_type","physics_tick","time_s","clock")} for e in lr["failure_events"]]==fe,"Earliest/tie-priority failure events differ")
                require(canonical(lr["label_rule_contract"])==lr["label_rule_sha256"]==meta["provenance_hashes"]["label_rule"],"Label-rule hash differs")
                require(abs(float(a["time_s"][-1])-5.5)<=1e-9,"Completed control horizon incomplete")
                entry.update(success=out["success"],failure_type=out["failure_type"],first_failure_time_s=out["first_failure_time_s"],label_arithmetic_metrics=metrics);complete_labels+=1
            else:require(meta["outcome"] is None and ledger[aid].get("outcome") is None,"Noncompleted slot acquired physical label")
            reason,feat=feature_prefix(a,out,meta["mapping"]["status"],meta["execution_status"]);entry.update(exclusion=reason,prefix=feat)
            if feat:prefix_hash.update(aid.encode()+bytes.fromhex(feat["feature_array_sha256"]))
        else:require(status in ("NOT_RUN","HARD_STOPPED_PARTIAL"),"Attempted/completed slot missing native marker")
        if status=="MAPPING_REFUSAL":require(entry["success"] is None and entry["exclusion"]=="mapping_or_setup_refusal","Refusal taxonomy differs")
        if status=="RUNTIME_QUARANTINE":require(ledger[aid]["reason_code"] in p["task_configuration"]["quarantine_reason_codes"] and entry["success"] is None and entry["exclusion"]=="runtime_or_unexecuted","Quarantine taxonomy differs")
        evaluated[aid]=entry
    manifest_rows={}
    for name in ("primary_schedule_before_task.json","category_split_manifest_before_task.json","catalog_seven_fold_manifest_before_task.json"):
        if (raw/name).is_file():manifest_rows[name]=read(raw/name)
    catalog_folds=[];trainer_folds=[];publication_order=None
    if manifest_rows:
        require(len(manifest_rows)==3,"Partial before-task manifest publication")
        sched=manifest_rows["primary_schedule_before_task.json"];sa={a["attempt_id"]:a for a in sched["attempts"]};require(set(sa)=={x["attempt_id"] for x in primary},"Primary schedule includes matched/missing IDs")
        contracts={canonical(c):c for c in sched["contracts"]};require(len(contracts)==30,"Primary per-object contracts differ")
        for row in primary:
            a=sa[row["attempt_id"]];require(a=={**row["training_identity"],"contract_sha256":a["contract_sha256"]} and a["contract_sha256"] in contracts,"Prospective source/group assignment changed")
            if (raw/row["attempt_id"]/"recording/attempt.json").is_file():observed_recording(raw/row["attempt_id"]/"recording",row,contracts[a["contract_sha256"]])
        five=manifest_rows["category_split_manifest_before_task.json"];verify_binding(five["schedule"]);require(five["selection_uses_outcomes"] is False and len(five["folds"])==5,"Trainer5 declaration differs")
        categories=sorted({x["category_id"] for x in primary},key=lambda s:hashlib.sha256(("native_category_order_v1:"+s).encode()).hexdigest())
        expected_five=[dict(fold_id="held_category_"+t,test_category=t,validation_category=categories[(i+1)%5],assignments={x["attempt_id"]:"test" if x["category_id"]==t else "validation" if x["category_id"]==categories[(i+1)%5] else "train" for x in primary}) for i,t in enumerate(categories)]
        require(five["folds"]==expected_five,"Trainer5 whole-category role policy differs");trainer_folds=fold_readiness(five["folds"],primary,evaluated)
        seven=manifest_rows["catalog_seven_fold_manifest_before_task.json"];require(seven["schema_version"]=="manosim_native_pilot_catalog_seven_folds_v1" and seven["catalog"]==p["catalog_binding"] and seven["source_fold_spec"]==p["catalog_folds_binding"] and seven["selection_uses_outcomes"] is False and seven["cohort"]=="primary120_only_matched1ms_excluded" and seven["trainer_five_category_manifest_unchanged"] is True and seven["seven_fold_training_API_implemented"] is False,"Catalog7 scope/identity differs")
        expected_seven=[]
        for f in foldspec["folds"]:
            membership={role:f[role+"_object_ids"] for role in ROLES};assignment={x["attempt_id"]:next(role for role in ROLES if x["object_id"] in membership[role]) for x in primary}
            expected_seven.append(dict(fold_id=f["fold_id"],generalization_claim=f["generalization_claim"],object_membership=membership,assignments=assignment))
        require(seven["folds"]==expected_seven,"Catalog7 object-role join differs");catalog_folds=fold_readiness(seven["folds"],primary,evaluated)
        positions={e["path"]:i for i,e in enumerate(events) if "event" not in e};laststatic=max(positions[str(raw/x["attempt_id"]/"source_chart_gate.npz")] for x in p["schedule"])
        manifestpos=[positions[str(raw/k)] for k in manifest_rows];require(min(manifestpos)>laststatic,"Manifests precede static preflight")
        after=[i for path,i in positions.items() if Path(path).name in ("physics_state.npz","partial_physics_state.npz","observables.npz")]
        require(not after or max(manifestpos)<min(after),"Manifest created after recorded task evidence")
        publication_order=dict(all135_static_artifacts_before_manifests=True,all_manifests_before_recorded_task_artifacts=True,scope="Journal order plus frozen source guards; no independent engine rerun.")
    collection=raw/"primary_collection_unqualified.json"
    if collection.is_file():
        co=read(collection);require(co["qualifications"]=={} and set(co["attempts"])=={x["attempt_id"] for x in primary},"Primary collection falsely qualifies/includes matched")
        for aid,b in co["attempts"].items():require(verify_binding(b)==raw/aid/"recording/attempt.json","Collection points outside primary canonical recording")
    if r["status"]=="COMPLETED":require(collection.is_file() and len(manifest_rows)==3 and len(evaluated)==135,"Completed collection missing denominators/manifests")
    def counts(rows):
        x=[evaluated[v["attempt_id"]] for v in rows];return dict(planned=len(x),statuses=dict(Counter(v["status"] for v in x)),quarantine_reason_counts=dict(Counter(v['reason_code'] for v in x if v.get('reason_code'))),failure_type_counts=dict(Counter(v['failure_type'] for v in x if v.get('failure_type'))),physical_success=sum(v["success"] is True for v in x),physical_failure=sum(v["success"] is False for v in x),null_outcomes=sum(v["success"] is None for v in x),eligible=sum(v["prefix"] is not None for v in x),eligible_successes=sum(v["prefix"] is not None and v["success"] is True for v in x),eligible_future_failures=sum(v["prefix"] is not None and v["success"] is False for v in x),exclusions=dict(Counter(v["exclusion"] for v in x if v["exclusion"])))
    coverage=[]
    for (category,size),oids in sorted(cells.items()):coverage.append(dict(category=category,size=size,official_object_ids=sorted(oids),primary=counts([v for v in primary if v["object_id"] in oids])))
    after=closure(root,protocol,p,freeze);require(sha(rp)==report_sha,"Report changed during readback")
    depth_only=bool(len(ledger)==135 and all(v['status']=='MAPPING_REFUSAL' and v.get('outcome') is None and {k for k,x in v['static_checks'].items() if not x}=={'depth_visual_alignment'} for v in ledger.values()))
    extra=dict(preserved_v1=prior,independent_camera_cast_readbacks=camera_rows,all135_depth_alignment_only_refusals=depth_only,collection_scope=p['collection_scope'],physical_RGBD_correspondence_calibrated=False,conditional_authored_asset_simulation_only=True)
    return dict(schema_version=SCHEMA,status="VERIFIED_SAVED_PILOT_COHORT_AND_PREFIX_READINESS" if r["status"]=="COMPLETED" else "VERIFIED_HARD_STOP_SAVED_PILOT_COHORT",experiment_status=r["status"],protocol=binding(protocol),report=binding(rp),freeze_commit=freeze,closure_before=before,closure_after=after,artifact_count=len(artifacts),creation_journal_events=len(events),publication_order=publication_order,actual_counts=r["actual_counts"],static_declared_check_counts=dict(static_counts),primary=counts(primary),matched_1ms=counts(matched),geometry_size_cells=coverage,catalog_seven_fold_readiness=catalog_folds,implemented_trainer_five_fold_readiness=trainer_folds,independent_completed_label_count=complete_labels,eligible_prefix_ordered_hash=prefix_hash.hexdigest(),attempts=list(evaluated.values()),scientific_training_allowed=False,training_gate="CLOSED_PENDING_INDEPENDENT_NATIVE_TASK_QUALIFICATION",admission_reasons=["Pilot collection qualifications remain empty; no exact native task qualification exists.","Catalog7 is a prospective readiness manifest; its fitting API is not implemented."],adequacy_guaranteed=False,operations=dict(engine_or_contact_calls=0,MANO_or_FK_calls=0,geometry_queries=0,physics_steps=0,fits=0),**extra,limits=["Labels are conditional on original authored HUG assets and this controller; RGBD physical correspondence is not calibrated. Saved label arithmetic relies on saved contact-force sums and measured state; mechanical/contact validation is a separate audit.","Static check counts preserve producer decisions; exact camera-cast arithmetic is independently replayed; other native/geometry operand replay is a separate mechanical audit.","Prefix eligibility and two-class role counts are software/data readiness, not admission, accuracy or independent object sample-size guarantees.","Two source RGBD observations and two HUG seeds per object; same object's views/seeds/perturbations stay together.","Original isolated-object-plus-table scene omits source-image clutter. No unseen-scene/camera or hardware generalization claim.","Matched1ms rows are numerical diagnostics, excluded from primary folds/training.","Null mapping/refusal/runtime/quarantine rows remain in allcase denominators and are never negative examples."])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--protocol",type=Path,required=True);ap.add_argument("--freeze",required=True);ap.add_argument("--report-sha256",required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    source=Path(__file__).resolve();source_bytes=source.read_bytes()
    try:result=verify(args.root.resolve(),args.protocol.resolve(),args.freeze,args.report_sha256)
    except BaseException as error:result=dict(schema_version=SCHEMA,status="INDEPENDENT_READBACK_HARD_STOP",failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()),scientific_training_allowed=False)
    result["verifier_source"]=dict(path=str(source),sha256=hashlib.sha256(source_bytes).hexdigest(),source_utf8=source_bytes.decode())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    print(json.dumps(dict(status=result["status"],receipt=binding(args.output))));return 0 if result["status"].startswith("VERIFIED_") else 1

if __name__=="__main__":raise SystemExit(main())
