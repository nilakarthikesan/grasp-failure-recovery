from pathlib import Path
import hashlib,json
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
R=Path('/private/tmp/compiled_hand_audit_20261015_v1/shadow/precise_spec_fix')
paths={n:W/'src/grasp_failure_prediction/evaluation'/n for n in ['pose_validation.py','runner.py']}
(R/'source_before_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},indent=2)+'\n')
s=paths['pose_validation.py'].read_text().replace('from dataclasses import dataclass\n', 'from dataclasses import dataclass\nimport hashlib\n')
start=s.index('def load_shadow_model(');end=s.index('\ndef _parse_xyz',start)
s=s[:start]+'''def load_shadow_spec(
    urdf_root: str | Path, *, floating_root: bool = False
) -> mujoco.MjSpec:
    """Parse the official Shadow URDF without a lossy compiled-XML round trip."""

    temporary = _mujoco_compatible_urdf(
        _shadow_urdf_path(urdf_root), floating_root=floating_root
    )
    try:
        tree = ET.parse(temporary)
        assets: dict[str, bytes] = {}
        for mesh in tree.getroot().findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename:
                continue
            source = Path(filename).resolve()
            # MuJoCo's URDF parser keys mesh assets by basename and scale even
            # with strippath=false. Distinct visual/collision files can share a
            # basename, so give every resolved file its own virtual filename.
            digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
            alias = f"gfp_{digest}_{source.name}"
            assets[alias] = source.read_bytes()
            mesh.set("filename", alias)
        return mujoco.MjSpec.from_string(
            ET.tostring(tree.getroot(), encoding="unicode"), assets=assets,
        )
    finally:
        temporary.unlink(missing_ok=True)


def load_shadow_model(
    urdf_root: str | Path, *, floating_root: bool = False
) -> mujoco.MjModel:
    """Compile the same official Shadow URDF used by Dex into MuJoCo."""

    return load_shadow_spec(urdf_root, floating_root=floating_root).compile()

''' +s[end:]
(R/'pose_validation.py').write_text(s)
s=paths['runner.py'].read_text().replace('import tempfile\n','import hashlib\n').replace('from .pose_validation import _body_id, load_shadow_model','from .pose_validation import _body_id, load_shadow_spec')
start=s.index('def _temporary_mjcf(');end=s.index('\n\nFLEXION_JOINTS',start)
s=s[:start]+'''def _build_tabletop_spec(
    urdf_root: str | Path,
    *,
    physics_timestep_s: float = 0.002,
    object_geometry: ResolvedObjectGeometry | None = None,
) -> mujoco.MjSpec:
    """Add the unchanged scene to the live Shadow spec at full precision."""

    spec = load_shadow_spec(urdf_root, floating_root=True)
    # URDF compilation fuses fixed links and mutates the spec's body tree.
    # Retain the resulting full-precision inertias before recompiling: otherwise
    # removed fixed-tip masses would disappear on the next compilation.
    hand = spec.compile()
    for body_id in range(1, hand.nbody):
        body = spec.body(mujoco.mj_id2name(hand, mujoco.mjtObj.mjOBJ_BODY, body_id))
        body.mass = float(hand.body_mass[body_id])
        body.ipos = hand.body_ipos[body_id]
        body.iquat = hand.body_iquat[body_id]
        body.inertia = hand.body_inertia[body_id]
        body.fullinertia = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
        body.explicitinertial = True
    spec.compiler.fusestatic = False
    # Hand visual geoms were already discarded by the URDF import. New object
    # visual geoms must remain, matching the former MJCF scene construction.
    spec.compiler.discardvisual = False
    spec.option.timestep = physics_timestep_s
    spec.option.gravity = [0.0, 0.0, -9.81]
    worldbody = spec.worldbody
    worldbody.add_light(
        name="key_light", pos=[0.0, -0.5, 1.2], dir=[0.0, 0.4, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    worldbody.add_camera(
        name="front", pos=[0.0, -0.7, 0.55],
        quat=[0.9238795, 0.3826834, 0.0, 0.0],
    )
    worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.4, 0.4, 0.025], pos=[0.0, 0.0, -0.025],
        rgba=[0.55, 0.45, 0.35, 1.0], friction=[1.0, 0.005, 0.0001],
    )
    if object_geometry is None:
        body = worldbody.add_body(name="object", pos=[0.0, 0.0, 0.03])
        body.add_freejoint(name="object_free")
        body.add_geom(
            name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025, 0.025, 0.025], mass=0.18,
            rgba=[0.15, 0.45, 0.8, 1.0], friction=[1.0, 0.005, 0.0001],
        )
    else:
        # Only the new object is parsed from XML. The already parsed hand never
        # passes through MuJoCo's precision-limited XML serializer. The existing
        # object importer verifies its assets and preserves native mesh frames.
        object_xml = ET.Element("mujoco", {"model": "gfp_object_attachment"})
        ET.SubElement(object_xml, "compiler", {"angle": "radian", "strippath": "false"})
        object_world = ET.SubElement(object_xml, "worldbody")
        append_object_geometry(object_xml, object_world, object_geometry)
        object_assets: dict[str, bytes] = {}
        for mesh in object_xml.findall("./asset/mesh"):
            path = Path(mesh.attrib["file"])
            digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
            alias = f"gfp_object_{digest}_{path.name}"
            object_assets[alias] = path.read_bytes()
            mesh.set("file", alias)
        object_spec = mujoco.MjSpec.from_string(
            ET.tostring(object_xml, encoding="unicode"), assets=object_assets,
        )
        spec.attach(object_spec, frame=worldbody.add_frame())
    return spec


def build_tabletop_model(
    urdf_root: str | Path,
    *,
    physics_timestep_s: float = 0.002,
    object_geometry: ResolvedObjectGeometry | None = None,
) -> mujoco.MjModel:
    """Compile the official hand and unchanged table/object scene directly."""

    return _build_tabletop_spec(
        urdf_root, physics_timestep_s=physics_timestep_s,
        object_geometry=object_geometry,
    ).compile()
''' +s[end:]
start=s.index('def add_shadow_actuation(');end=s.index('\ndef _joint_qpos_address',start)
s=s[:start]+'''def add_shadow_actuation(scene_spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Add the existing force-limited servos without serializing the hand.

    A live specification is required: recovering it from an arbitrary compiled
    model through ``mj_saveLastXML`` loses precision and depends on global XML
    parser state. The caller's specification is left unchanged.
    """

    if not isinstance(scene_spec, mujoco.MjSpec):
        raise TypeError("add_shadow_actuation requires the live tabletop MjSpec")
    spec = scene_spec.copy()
    base = spec.compile()
    root_joint = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
    if root_joint < 0 or base.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise RetargetingError("Shadow scene requires its original free world_joint")
    address = int(base.jnt_qposadr[root_joint])
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 10
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_body(
        name="arm_target", mocap=True,
        pos=base.qpos0[address : address + 3],
        quat=base.qpos0[address + 3 : address + 7],
    )
    spec.add_equality(
        name="ideal_arm", type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY, name1="arm_target", name2="forearm",
        data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        solref=[0.01, 1.0],
    )
    for joint in spec.joints:
        name = joint.name
        if not name or joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        effort = abs(float(joint.actfrcrange[1]))
        if not name.startswith("WR"):
            effort = min(effort, 1.0)
        actuator = spec.add_actuator(
            name=f"servo_{name}", target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=True, ctrlrange=joint.range,
            forcelimited=True, forcerange=[-effort, effort],
        )
        actuator.set_to_position(kp=50.0 if name.startswith("WR") else 10.0, kv=0.3)
    return spec.compile()

''' +s[end:]
s=s.replace('''        self.model = build_tabletop_model(
            urdf_root, physics_timestep_s=physics_timestep_s, object_geometry=object_geometry
        )''','''        self._scene_spec = _build_tabletop_spec(
            urdf_root, physics_timestep_s=physics_timestep_s, object_geometry=object_geometry
        )
        self.model = self._scene_spec.compile()''')
s=s.replace('self.model = add_shadow_actuation(self.model)','self.model = add_shadow_actuation(self._scene_spec)')
assert 'mj_saveLastXML(' not in s and '_temporary_mjcf' not in s
(R/'runner.py').write_text(s)
