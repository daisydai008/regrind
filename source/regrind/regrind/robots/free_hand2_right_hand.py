"""Isolated floating Hand2 RIGHT configuration; not a task or training asset."""
import importlib.util
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

_spec = importlib.util.spec_from_file_location(
    "hand2_right_contract", Path(__file__).with_name("hand2_right_contract.py"))
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


QUERY_FRAME_NAMES = tuple(f"r_{finger}_tip" for finger in
                          ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")) + ("r_wrist_imu_frame",)
QUERY_FRAME_MASS_KG = 1e-7
QUERY_FRAME_DIAGONAL_INERTIA = (1e-7, 1e-7, 1e-7)
QUERY_FRAME_SOURCE_USD = "hand2/hand2_2/body/usd/right/wujihand2_2.usd"
QUERY_FRAME_SOURCE_SHA256 = "ea9e12d7f0d96f33b91c95e193c87ecea91cbe71f2deabe5c135c1d77f52aa3f"
PLACEHOLDER_WARNING = "Official vendor numerical placeholders, not measured physical-link mass/inertia; not controller calibration."


def expected_mass_contract():
    """Derive physical masses from the pinned local URDF; no vendor/runtime import."""
    contract.validate_asset()
    root = ET.parse(contract.C.ROBOT_URDF).getroot()
    links = root.findall("link")
    names = [link.get("name") for link in links]
    if len(names) != 33 or len(set(names)) != 33:
        raise ValueError("Expected exactly 33 unique Hand2 links")
    physical, frames = {}, {}
    for link in links:
        name = link.get("name")
        if name in QUERY_FRAME_NAMES:
            if any(link.find(tag) is not None for tag in ("inertial", "visual", "collision")):
                raise ValueError(f"Query frame is not massless/geometry-free in pinned URDF: {name}")
            parents = [j for j in root.findall("joint") if j.find("child").get("link") == name]
            if (len(parents) != 1 or parents[0].get("type") != "fixed"
                    or parents[0].find("parent").get("link") not in set(names) - set(QUERY_FRAME_NAMES)):
                raise ValueError(f"Query frame must have one fixed physical parent: {name}")
            frames[name] = QUERY_FRAME_MASS_KG
        else:
            mass = link.find("inertial/mass")
            value = float(mass.get("value")) if mass is not None else 0.0
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Missing positive physical mass: {name}")
            physical[name] = value
    if set(frames) != set(QUERY_FRAME_NAMES) or len(physical) != 27:
        raise ValueError("Expected exactly 27 physical links and six query frames")
    if not math.isclose(math.fsum(physical.values()), 0.6878, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Pinned physical mass sum changed")
    return {"physical_mass_kg": physical, "query_frame_mass_kg": frames,
            "expected_total_mass_kg": math.fsum([*physical.values(), *frames.values()]),
            "placeholder_diagonal_inertia": list(QUERY_FRAME_DIAGONAL_INERTIA),
            "source": {"label": "official_vendor_pin", "model_pin": contract.C.MODEL_PIN,
                       "usd_path": QUERY_FRAME_SOURCE_USD, "usd_sha256": QUERY_FRAME_SOURCE_SHA256},
            "placeholder_warning": PLACEHOLDER_WARNING}


def apply_query_frame_mass_defaults(stage, prim_path, usd_modules=None):
    """Before reset, author only the six nonphysical frames, never sensor pads."""
    expected = expected_mass_contract()  # Fail before any USD authoring.
    contract.robot_path(prim_path)
    if usd_modules is None:
        from pxr import Usd, UsdPhysics, Gf
    else:
        Usd, UsdPhysics, Gf = usd_modules
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid() or root.IsInstance() or root.IsInstanceProxy():
        raise ValueError(f"Missing/instance Robot: {prim_path}")
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    frames = {}
    for name in QUERY_FRAME_NAMES:
        matches = [p for p in prims if p.GetName() == name]
        if len(matches) != 1:
            raise ValueError(f"Missing/duplicate query frame: {name}")
        prim = matches[0]
        path = str(prim.GetPath())
        if (not path.startswith(prim_path + "/") or prim.IsInstance() or prim.IsInstanceProxy()
                or not prim.HasAPI(UsdPhysics.RigidBodyAPI)):
            raise ValueError(f"Query frame is not an editable own-robot rigid body: {path}")
        ancestor = prim.GetParent()
        while ancestor.IsValid() and ancestor != root:
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI) or ancestor.IsInstance() or ancestor.IsInstanceProxy():
                raise ValueError(f"Nested/instance query frame rejected: {path}")
            ancestor = ancestor.GetParent()
        if ancestor != root:
            raise ValueError(f"Query frame outside Robot: {path}")
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if child.HasAPI(UsdPhysics.CollisionAPI) or (child != prim and child.HasAPI(UsdPhysics.RigidBodyAPI)):
                raise ValueError(f"Physical geometry/nested body on query frame: {path}")
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if mass is not None and not any(math.isclose(mass, v, rel_tol=0, abs_tol=5e-12)
                                            for v in (0.0, 1.0, QUERY_FRAME_MASS_KG)):
                raise ValueError(f"Unexpected existing physical mass on query frame: {path}: {mass}")
        frames[name] = prim
    records = {}
    for name, prim in frames.items():
        api = UsdPhysics.MassAPI.Apply(prim)
        if not api or not api.CreateMassAttr().Set(QUERY_FRAME_MASS_KG):
            raise RuntimeError(f"Cannot author query-frame mass: {name}")
        if not api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*QUERY_FRAME_DIAGONAL_INERTIA)):
            raise RuntimeError(f"Cannot author query-frame inertia: {name}")
        records[name] = {"prim_path": str(prim.GetPath()), "mass_kg": api.GetMassAttr().Get(),
                         "diagonal_inertia": list(api.GetDiagonalInertiaAttr().Get())}
    return {"frames": records, "source": expected["source"], "placeholder_warning": PLACEHOLDER_WARNING}


def check_runtime_masses(robot):
    """Check live PhysX masses in every environment, not cached default_mass."""
    expected = expected_mass_contract()
    masses = {**expected["physical_mass_kg"], **expected["query_frame_mass_kg"]}
    names = list(robot.body_names)
    if len(names) != 33 or len(set(names)) != 33 or set(names) != set(masses):
        raise ValueError("Actual PhysX body names must match all 33 unique Hand2 links")
    if not hasattr(robot.root_physx_view, "get_masses"):
        raise RuntimeError("Required live PhysX get_masses unavailable")
    tensor = robot.root_physx_view.get_masses()
    if tuple(tensor.shape) != (robot.num_instances, 33) or robot.num_instances < 1:
        raise ValueError(f"Unexpected live PhysX mass tensor shape: {tensor.shape}")
    rows = tensor.detach().cpu().tolist()
    records, totals = [], []
    for env_index, row in enumerate(rows):
        for name, value in zip(names, row):
            placeholder = name in expected["query_frame_mass_kg"]
            if (not math.isfinite(value) or value <= 0
                    or not math.isclose(value, masses[name], rel_tol=0 if placeholder else 1e-5,
                                        abs_tol=5e-12 if placeholder else 1e-8)):
                raise ValueError(f"Actual PhysX mass mismatch: env={env_index} {name}: {value} != {masses[name]}")
        total = math.fsum(row)
        if not math.isclose(total, 0.6878006, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Actual PhysX total mass mismatch: env={env_index}: {total}")
        records.append(dict(zip(names, row)))
        totals.append(total)
    return {"verification_state": "passed", "evidence": "live_root_physx_view.get_masses",
            "body_mass_kg_per_env": records, "total_mass_kg_per_env": totals, **expected}


def _resolve_collision_links(stage, prim_path, Usd, UsdPhysics):
    contract.robot_path(prim_path)
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"Robot prim missing: {prim_path}")
    _, names, _ = contract.model_layout()
    prims = list(Usd.PrimRange(root))
    result = {}
    for name in names:
        matches = [p for p in prims if p.GetName() == name]
        if len(matches) != 1:
            raise ValueError(f"Missing/duplicate imported collision link: {name}")
        link = matches[0]
        path = str(link.GetPath())
        if not path.startswith(prim_path + "/"):
            raise ValueError(f"Cross-robot link rejected: {path}")
        if link.IsInstanceProxy() or link.IsInstance():
            raise ValueError(f"Cannot author filters on instance: {path}")
        if not link.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Collision link is not an actual rigid body: {path}")
        # Instance-proxy meshes are read-only; filters stay on the actual rigid link.
        colliders = [p for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.CollisionAPI)]
        if not colliders or any(not str(p.GetPath()).startswith(path + "/")
                                and str(p.GetPath()) != path for p in colliders):
            raise ValueError(f"Collision geometry missing or outside link: {path}")
        for collider in colliders:
            owner = collider
            while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if not owner.IsValid() or owner != link:
                raise ValueError(f"Collision geometry rigid body owner mismatch: {collider.GetPath()}")
        result[name] = link
    return root, result


def inspect_collision_filters(stage, prim_path, usd_modules=None):
    """Read authored relationships, NOT a PhysX contact/dynamics probe."""
    if usd_modules is None:
        from pxr import Usd, UsdPhysics
    else:
        Usd, UsdPhysics = usd_modules
    root, links = _resolve_collision_links(stage, prim_path, Usd, UsdPhysics)
    _, _, pairs = contract.model_layout()
    expected = {tuple(sorted((str(links[a].GetPath()), str(links[b].GetPath())))) for a, b in pairs}
    actual = set()
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                source, target = str(prim.GetPath()), str(target)
                if not source.startswith(prim_path + "/") or not target.startswith(prim_path + "/"):
                    raise ValueError(f"Cross-robot/object filter rejected: {source} -> {target}")
                pair = tuple(sorted((source, target)))
                if pair not in expected:
                    raise ValueError(f"Unexpected filter relationship: {pair}")
                actual.add(pair)
    if actual != expected:
        raise ValueError(f"Authored filters incomplete: missing {sorted(expected - actual)}")
    return {"evidence": "authored_UsdPhysics_FilteredPairsAPI_only",
            "expected_link_pairs": [list(pair) for pair in pairs],
            "actual_prim_pairs": [list(pair) for pair in sorted(actual)],
            "inherent_parent_filters": "PhysX may additionally suppress adjacent articulated links",
            "contact_or_dynamics_verified": False}


def apply_collision_filters(stage, prim_path, usd_modules=None):
    if usd_modules is None:
        from pxr import Usd, UsdPhysics
    else:
        Usd, UsdPhysics = usd_modules
    _, links = _resolve_collision_links(stage, prim_path, Usd, UsdPhysics)
    for a, b in contract.model_layout()[2]:
        api = UsdPhysics.FilteredPairsAPI.Apply(links[a])
        if not api or not api.CreateFilteredPairsRel().AddTarget(links[b].GetPath()):
            raise RuntimeError(f"Cannot author collision filter: {a}, {b}")
    return inspect_collision_filters(stage, prim_path, (Usd, UsdPhysics))


def prepare_import_aliases(asset_path, usd_dir):
    """Own a new cache and copy pinned bytes; never rewrite the canonical asset."""
    manifest = contract.validate_asset()
    source = contract.safe_path(asset_path)
    if source != contract.safe_path(contract.C.ROBOT_URDF):
        raise ValueError("Only the pinned Hand2 URDF may be aliased")
    cache = contract.new_path(usd_dir)
    bundle = cache / "input_assets"
    tree = ET.parse(source)
    root = tree.getroot()
    original_name = root.get("name")
    if original_name != "wujihand2_2-right":
        raise ValueError("Unexpected pinned robot name")
    root.set("name", "wujihand2_2_right")
    payload = manifest["generated_files_sha256"]
    records = []
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        relative = Path(filename)
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3
                or relative.parts[:2] not in (("meshes", "visual"), ("meshes", "collision"))
                or relative.as_posix() != filename or filename not in payload):
            raise ValueError(f"Unpinned or escaping mesh path: {filename}")
        src = contract.safe_path(source.parent / relative)
        if not src.is_relative_to(source.parent):
            raise ValueError(f"Escaping source mesh: {src}")
        alias = relative.with_name("mesh_" + relative.name)
        if alias.suffix != ".obj" or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", alias.stem):
            raise ValueError(f"Invalid USD mesh identifier: {alias.stem}")
        records.append({"source_path": str(src), "source_relative_path": filename,
                        "source_sha256": payload[filename],
                        "generated_path": str(bundle / alias),
                        "generated_relative_path": alias.as_posix()})
        mesh.set("filename", alias.as_posix())
    if (len(records) != 54 or len({r["source_path"] for r in records}) != 54
            or set(payload) != {source.name, *(r["source_relative_path"] for r in records)}):
        raise ValueError("Expected exactly the pinned URDF and 54 distinct OBJ meshes")
    # All preflight checks precede exclusive directory ownership. Failed imports
    # leave their caller-owned evidence in place; there is no retry or cleanup.
    cache.mkdir()
    bundle.mkdir()
    for record in records:
        target = Path(record["generated_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target = contract.new_path(target)
        with target.open("xb") as stream:
            stream.write(Path(record["source_path"]).read_bytes())
        record["generated_sha256"] = contract.C.sha256(target)
        if record["generated_sha256"] != record["source_sha256"]:
            raise ValueError(f"Source changed while copying: {record['source_path']}")
    generated = bundle / source.name
    with generated.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)
    alias_manifest = {
        "import_only": True, "canonical_model": False, "approved_for_training": False,
        "source_urdf_path": str(source), "source_urdf_sha256": contract.ASSET_URDF_SHA256,
        "generated_urdf_path": str(generated), "generated_urdf_sha256": contract.C.sha256(generated),
        "source_manifest_sha256": contract.C.MANIFEST_SHA256,
        "original_robot_name": original_name, "cached_robot_name": root.get("name"),
        "name_policy": "robot hyphen to underscore; prefix each OBJ basename with mesh_; "
                       "preserve visual/collision directories and all link/joint names",
        "meshes": records,
    }
    with (bundle / "alias_manifest.json").open("x") as stream:
        json.dump(alias_manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return generated


def spawn_hand2_right(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Author exact filters immediately after import, before first physics reset."""
    import isaaclab.sim as sim_utils
    import omni.usd

    contract.robot_path(prim_path)
    contract.new_path(cfg.usd_dir)
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"Refusing to reuse an existing Robot prim: {prim_path}")
    alias_urdf = prepare_import_aliases(cfg.asset_path, cfg.usd_dir)
    import_cfg = cfg.replace(asset_path=str(alias_urdf))
    prim = sim_utils.spawn_from_urdf(
        prim_path, import_cfg, translation=translation, orientation=orientation, **kwargs)
    apply_query_frame_mass_defaults(stage, prim_path)
    apply_collision_filters(stage, prim_path)
    return prim


def make_hand2_right_cfg(prim_path, usd_dir):
    """Caller owns a never-used cache path; constructing config writes nothing."""
    contract.robot_path(prim_path)
    usd_dir = contract.new_path(usd_dir)
    contract.validate_asset()
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    values = contract.PARAMETERS
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            func=spawn_hand2_right,
            asset_path=str(contract.C.ROBOT_URDF), usd_dir=str(usd_dir),
            fix_base=False, root_link_name="r_mount", merge_fixed_joints=False,
            self_collision=True,
            # Validation-only choice: permit per-link USD API authorship.
            make_instanceable=False,
            # IsaacLab 2.3 has no import_inertia_tensor configuration field.
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False, enabled_self_collisions=True),
        ),
        actuators={"fingers": ImplicitActuatorCfg(
            joint_names_expr=list(contract.JOINT_NAMES),
            stiffness={n: p["kp"] for n, p in values.items()},
            damping={n: p["kv"] for n, p in values.items()},
            effort_limit_sim={n: p["effort"] for n, p in values.items()},
            velocity_limit_sim={n: p["velocity"] for n, p in values.items()},
            armature={n: p["armature"] for n, p in values.items()},
        )},
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={n: 0.0 for n in values}),
        soft_joint_pos_limit_factor=1.0,
    )


def spawn_hand2_right_training(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Import once, then let Isaac's clone decorator remap the USD relationships."""
    if prim_path != "/World/envs/env_.*/Robot":
        raise ValueError("Training spawn accepts only /World/envs/env_.*/Robot")
    if cfg.make_instanceable:
        raise ValueError("Hand2 filtered training clones must not be instanceable")
    import isaaclab.sim as sim_utils
    import omni.usd

    # The concrete spawner alone owns the new cache. Never import once per env.
    prim = sim_utils.clone(spawn_hand2_right)(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = omni.usd.get_context().get_stage()
    for path in sim_utils.find_matching_prim_paths(prim_path):
        inspect_collision_filters(stage, path)
    return prim


def make_hand2_right_training_cfg(prim_path, usd_dir):
    """Explicit training adapter; no cache creation or generic-regex permission."""
    if prim_path not in ("{ENV_REGEX_NS}/Robot", "/World/envs/env_.*/Robot"):
        raise ValueError("Expected the exact Hand2 training environment namespace")
    cfg = make_hand2_right_cfg("/World/envs/env_0/Robot", usd_dir)
    return cfg.replace(
        prim_path=prim_path,
        spawn=cfg.spawn.replace(func=spawn_hand2_right_training, copy_from_source=True),
    )
