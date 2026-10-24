"""Isolated floating Hand2 RIGHT configuration; not a task or training asset."""
import importlib.util
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

_spec = importlib.util.spec_from_file_location(
    "hand2_right_contract", Path(__file__).with_name("hand2_right_contract.py"))
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


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
