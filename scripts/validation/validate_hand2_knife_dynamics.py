#!/usr/bin/env python3
"""Local one-environment Knife force audit, NOT POLICY / training approval.

Default / --check-only is read-only and simulator-free. Only --run with both
explicit existing-record and fresh-video paths can start Isaac. No task is added.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
REPO = ROOT / "repos/regrind"
SOURCE = ROOT / "data/tactidex/retargeted/wujihand2_diagnostic/knife_parent_weld_meshtip_table002_v1"
DEMO = ROOT / "data/tactidex/regrind_demo_meshtip_hand2_right_v1/TactiDex_Cut_S0H01A0022T1_R/demo_30fps.h5"
URDF = ROOT / "data/tactidex/assets/knife/knife.urdf"
CONTRACT = ROOT / ".pi/skills/regrind-right-hand/references/hand2-right-eight.yaml"
RHO = REPO / "scripts/retarget_hand_object.py"
ARTIFACTS = ROOT / "REGRIND/artifacts/hand2_right/mesh_tip_ppo_check_v1"
VIDEO = ARTIFACTS / "knife_zero_residual_force_audit_attempt3.mp4"
RECORD = ARTIFACTS / "run.json"
KEY = "knife_isaac_force_audit_attempt3"
PINS = {
    SOURCE / "retargeted_120fps.h5": "77f7337d92688ca154d67dcd905c334ae96ea9b06b07214aee134cc18ba05037",
    SOURCE / "metrics.json": "ec7197055a8309e0bd4286e4a8056607b01ebf6a0920154b111af41acd47476b",
    SOURCE / "run.json": "58e301418ed9a9b619d7f75b84739533b21b4395609449ba637bb0cb3c9ad61c",
    DEMO: "87f70b29fc1cda8e0c2c340d6e483a9d96124ef004a9612b0aa6ac0e8fdb24d4",
    URDF: "3acf5dafbdd1a7a4a2848fc3f0e6cf00d3a552eaa9dcbead72d75d63c780a610",
    CONTRACT: "ac39f690cdb425982ab862dde07a0f6da9e66d04aa73fcf7af14b7c8d947324d",
    RHO: "86cc71fc8d228f3c5102ea4a5abf7b8ef31b104eb400a90d8aa71251faa546cf",
}
MAPPING_HASH = "c8b6dade2a775bc2140a55d82a606a43ee19e94245b6a80927b0930eed004cf5"
WINDOWS = ((133, 161), (192, 225), (249, 280), (297, 333), (354, 392))
CROP = slice(528, 2241)
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
SENSORS = tuple(f"contact_hand2_{f}_table" for f in FINGERS)
CONTROLS = 240


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_pins(pins):
    result = {}
    for path, expected in pins.items():
        path = Path(path)
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Source pin mismatch: {path}")
        result[str(path)] = expected
    return result


def old_allowed(step, dt=1 / 30):
    time_s = step * dt
    return any(a / 30 - 1e-9 <= time_s <= b / 30 + 1e-9 for a, b in WINDOWS)


def interpolate_source():
    """Execute only the two pinned RHO pure interpolation functions, no Drake import."""
    import h5py
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    verify_pins({RHO: PINS[RHO]})
    tree = ast.parse(RHO.read_text())
    names = {"interpolate_poses", "interpolate_demo"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Missing RHO interpolation functions")
    ns = {"np": np, "R": Rotation, "Slerp": Slerp}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RHO), "exec"), ns)
    with h5py.File(DEMO, "r") as h:
        arrays = {k: h[k][()] for k in h if isinstance(h[k], h5py.Dataset)}
    shapes = {"mano_joint_coords": (752, 21, 3), "obj_pos": (752, 3), "obj_quat": (752, 4),
              "wrist_pos": (752, 3), "wrist_quat": (752, 4)}
    for k, shape in shapes.items():
        if arrays[k].shape != shape or not np.isfinite(arrays[k]).all():
            raise ValueError(f"Source demo shape/finite: {k}")
    wrist = np.c_[Rotation.from_quat(arrays["wrist_quat"]).as_quat(scalar_first=True), arrays["wrist_pos"]]
    objects = np.c_[arrays["obj_quat"], arrays["obj_pos"]]
    full = ns["interpolate_demo"](wrist, arrays["mano_joint_coords"], objects, None, 4)
    if full[3] is not None or not np.array_equal(full[1][::4], arrays["mano_joint_coords"]):
        raise ValueError("Human native samples changed")
    if full[1].shape != (3005, 21, 3) or not np.array_equal(full[2][::4, 4:], arrays["obj_pos"]):
        raise ValueError("Full native interpolation mismatch")
    if not np.array_equal(full[1][CROP][::4], arrays["mano_joint_coords"][132:561]):
        raise ValueError("Crop/native source frame mismatch")
    return full


def precheck():
    import h5py
    import numpy as np
    import yaml
    pins = dict(PINS)
    verify_pins(pins)
    source_run = json.loads((SOURCE / "run.json").read_text())
    # The pinned run supplies the complete original input/model and mapping-code hashes.
    for group in ("input_model_sha256", "code_sha256"):
        pins.update({ROOT / k: v for k, v in source_run[group].items()})
    snapshot = verify_pins(pins)
    if source_run["approved_for_training"] is not False or source_run["quality_pass"] is not True:
        raise ValueError("Unexpected diagnostic status")
    metrics = json.loads((SOURCE / "metrics.json").read_text())
    if metrics["quality_pass"] is not True or metrics["frames_120fps"] != 3005:
        raise ValueError("Unexpected source metrics")
    contract = yaml.safe_load(CONTRACT.read_text())
    case, = [c for c in contract["cases"] if c["object_name"] == "knife"]
    if (case["training"]["native_window_30fps_closed"] != [132, 560]
            or case["training"]["frames_120fps"] != 1713 or case["table_height_m"] != .002):
        raise ValueError("Frozen Knife contract changed")
    keypoints = ROOT / case["keypoints"]["path"]
    verify_pins({keypoints: case["keypoints"]["sha256"]})
    points = np.load(keypoints, allow_pickle=False)
    if points.shape != (50, 3) or not np.isfinite(points).all():
        raise ValueError("Knife requires its actual 50 keypoints, not Bowl keypoints")
    root = ET.parse(URDF).getroot()
    links = root.findall("link")
    if len(links) != 1 or links[0].get("name") != "bottom" or root.findall("joint"):
        raise ValueError("Knife must be a single rigid bottom link")
    hulls = links[0].findall("collision/geometry/mesh")
    meshes = [e.get("filename") for e in root.findall(".//mesh")]
    if len(hulls) != 6 or len(set(meshes)) != 7:
        raise ValueError("Expected six hulls and one visual OBJ")
    missing = set()
    for name in meshes:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Nonlocal mesh reference")
        mesh = URDF.parent / name
        if str(mesh) not in snapshot:
            raise ValueError("Unpinned mesh")
        for line in mesh.read_text().splitlines():
            if line.startswith("mtllib "):
                material = mesh.parent / line.split(maxsplit=1)[1]
                if material.exists():
                    raise ValueError("Unexpected material, do not repair visual assets")
                missing.add(str(material.relative_to(ROOT)))
    if missing != {"data/tactidex/assets/knife/3DModel.mtl"}:
        raise ValueError("Unexpected missing visual dependencies")
    full = interpolate_source()
    with h5py.File(SOURCE / "retargeted_120fps.h5", "r") as h:
        if (hashlib.sha256(h.attrs["mapping"].encode()).hexdigest() != MAPPING_HASH
                or bool(h.attrs["approved_for_training"]) or h.attrs["root_frame"] != "r_mount"
                or h.attrs["fps"] != 120 or h.attrs["quaternion_order"] != "wxyz"):
            raise ValueError("H5 frozen metadata/mapping mismatch")
        for key, tail in {"robot_pos": (3,), "robot_quat": (4,), "robot_joints": (20,),
                          "robot_keypoints": (21, 3), "mano_joint_coords": (21, 3),
                          "object_pos": (3,), "object_quat": (4,)}.items():
            if h[key].shape != (3005, *tail) or not np.isfinite(h[key][()]).all():
                raise ValueError(f"Full H5 shape/finite mismatch: {key}")
        if not np.array_equal(h["mano_joint_coords"][()], full[1]):
            raise ValueError("Retargeted human source changed")
        if not np.array_equal(h["object_pos"][()], full[2][:, 4:]):
            raise ValueError("Retargeted object translations changed")
        if not np.allclose(np.abs(np.sum(h["object_quat"][()] * full[2][:, :4], axis=1)), 1, atol=1e-6):
            raise ValueError("Retargeted object rotations changed")
    return {"source_hashes_before": snapshot, "keypoints": str(keypoints), "keypoint_count": 50,
            "object_meshes": meshes, "object_collision_hulls": 6,
            "missing_visual_dependencies": sorted(missing), "visual_fidelity_validated": False,
            "mapping_sha256": MAPPING_HASH, "crop_120fps_half_open": [528, 2241],
            "native_frames": 1713, "control_frames": 429, "source_human_native_exact": True}


def stage_inputs(directory, checked):
    """Only caller-owned TemporaryDirectory receives data/cache; source bytes stay intact."""
    import h5py
    import numpy as np
    directory = Path(directory)
    full = interpolate_source()
    with h5py.File(SOURCE / "retargeted_120fps.h5", "r") as src, h5py.File(directory / "retargeted_120fps.h5", "x") as dst:
        dst.attrs.update(src.attrs)
        for key, value in src.items():
            data = value[CROP] if value.shape and value.shape[0] == 3005 else value[()]
            created = dst.create_dataset(key, data=data)
            created.attrs.update(value.attrs)
    # Preserve all source metadata and interpolate only the temporal arrays.
    with h5py.File(DEMO, "r") as src, h5py.File(directory / "demo_interp_120fps.h5", "x") as dst:
        dst.attrs.update(src.attrs)
        arrays = {"mano_joint_coords": full[1], "obj_pos": full[2][:, 4:],
                  "obj_quat": full[2][:, :4], "wrist_pos": full[0][:, 4:],
                  "wrist_quat": full[0][:, [1, 2, 3, 0]]}
        for key, value in src.items():
            if isinstance(value, h5py.Group):
                src.copy(value, dst, name=key)
                continue
            created = dst.create_dataset(key, data=arrays[key][CROP] if key in arrays else value[()])
            created.attrs.update(value.attrs)
    with h5py.File(directory / "retargeted_120fps.h5", "r") as h:
        if len(h["robot_pos"]) != 1713 or not np.array_equal(h["mano_joint_coords"][::4], full[1][CROP][::4]):
            raise ValueError("Staged crop mismatch")
    for name in ["knife.urdf", *checked["object_meshes"]]:
        target = directory / "object" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(URDF.parent / name, target)
        if sha256(target) != sha256(URDF.parent / name):
            raise ValueError("Staged object bytes changed")
    # Preserve the original bundle; only the temporary importer input uses a USD-safe basename.
    original = directory / "object/knife.urdf"
    source_mesh = directory / "object/3DModel.obj"
    alias_mesh = directory / "object/mesh_visual.obj"
    importer = directory / "object/knife_import.urdf"
    old, new = b'filename="3DModel.obj"', b'filename="mesh_visual.obj"'
    original_bytes = original.read_bytes()
    if original_bytes.count(old) != 1:
        raise ValueError("Expected exactly one original visual filename reference")
    importer_bytes = original_bytes.replace(old, new, 1)
    original_xml, importer_xml = ET.fromstring(original_bytes), ET.fromstring(importer_bytes)
    visuals = importer_xml.findall("link/visual/geometry/mesh")
    if len(visuals) != 1 or visuals[0].get("filename") != "mesh_visual.obj":
        raise ValueError("Importer alias must change only the visual mesh filename")
    for tag, count in (("inertial", 1), ("collision", 6)):
        before, after = original_xml.findall(f"link/{tag}"), importer_xml.findall(f"link/{tag}")
        if len(before) != count or [ET.tostring(e) for e in before] != [ET.tostring(e) for e in after]:
            raise ValueError(f"Importer alias changed {tag}")
    visuals[0].set("filename", "3DModel.obj")
    if ET.tostring(importer_xml) != ET.tostring(original_xml):
        raise ValueError("Importer XML changed beyond the visual filename")
    shutil.copyfile(source_mesh, alias_mesh)
    if alias_mesh.read_bytes() != source_mesh.read_bytes():
        raise ValueError("Visual mesh alias bytes changed")
    with importer.open("xb") as stream:
        stream.write(importer_bytes)
    shutil.copyfile(checked["keypoints"], directory / "keypoints.npy")
    return {"scope": "temporary importer-only visual filename alias; no source/material repair",
            "reason": "digit-leading 3DModel basename produces invalid USD SdfPath",
            "source_urdf": str(URDF), "source_urdf_sha256": sha256(URDF),
            "staged_original_urdf_sha256": sha256(original),
            "importer_urdf": "object/knife_import.urdf", "importer_urdf_sha256": sha256(importer),
            "replacement": {"old": old.decode(), "new": new.decode(), "count": 1},
            "source_mesh": str(URDF.parent / "3DModel.obj"),
            "source_mesh_sha256": sha256(URDF.parent / "3DModel.obj"),
            "staged_original_mesh_sha256": sha256(source_mesh),
            "alias_mesh": "object/mesh_visual.obj", "alias_mesh_sha256": sha256(alias_mesh),
            "inertial_and_six_collisions_unchanged": True, "only_visual_filename_changed": True,
            "visual_fidelity_validated": False, "missing_material_unrepaired": "3DModel.mtl"}


def make_config(directory):
    from isaaclab.utils import configclass
    from regrind.tasks.manager_based.dexterous.config.wujihand2.hand2_bowl_s_env_cfg import Hand2BowlSEnvCfg
    from regrind.tasks.manager_based.dexterous.mdp.hand2_supported_root import Hand2SupportedRootActionCfg
    from regrind.tasks.manager_based.dexterous.mdp.terminations import hit_table_outside_window
    from regrind.objects.tactidex_objects import KNIFE_OBJECT_CFG

    @configclass
    class LocalKnifeForceAuditCfg(Hand2BowlSEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 1
            self.episode_length_s = 429 / 30
            self.scene.robot.spawn.rigid_props.disable_gravity = False
            self.scene.object = KNIFE_OBJECT_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Object", spawn=KNIFE_OBJECT_CFG.spawn.replace(
                    asset_path=str(directory / "object/knife_import.urdf"), usd_dir=str(directory / "object_usd")))
            self.scene.object.spawn.rigid_props.disable_gravity = False
            self.scene.table.init_state.pos = (-.15, 0., .002 - self.scene.table.spawn.size[2] / 2)
            motion = self.commands.motion
            motion.demo_path = str(directory / "demo_interp_120fps.h5")
            motion.retargeted_traj_path = str(directory / "retargeted_120fps.h5")
            motion.keypoint_paths = {"bottom": str(directory / "keypoints.npy")}
            motion.object_pos_lower_bound = (-.443, -.277, 0.)
            motion.object_pos_upper_bound = (-.070, .226, .395)
            old = self.actions.root_pose
            self.actions.root_pose = Hand2SupportedRootActionCfg(
                asset_name=old.asset_name, body_name="r_mount", base_action_source=old.base_action_source,
                command_name=old.command_name, raw_clip=old.raw_clip,
                scale_pos=old.scale_pos, scale_rot=old.scale_rot,
                root_controller_profile="com_bounded_v3", kp_pos=300., kd_pos=30., kp_rot=3., kd_rot=.02)
            self.actions.joint_pos.scale *= 3.0
            self.terminations.hit_table.func = hit_table_outside_window
            self.terminations.hit_table.params.update(
                command_name="motion", allowed_windows_s=[(a / 30, b / 30) for a, b in WINDOWS])

    cfg = LocalKnifeForceAuditCfg()
    if (cfg.decimation != 4 or cfg.sim.dt != 1 / 120 or tuple(cfg.sim.gravity) != (0., 0., -9.81)
            or cfg.terminations.hit_table.params["force_threshold"] != 10.
            or cfg.terminations.hit_table.params["sensor_names"] != list(SENSORS)):
        raise ValueError("Unexpected inherited force/physics contract")
    return cfg


def install_capture(env, rows):
    """Wrap compute, never a term: every control is observed BEFORE normal resets."""
    import torch
    from isaaclab.utils.math import quat_error_magnitude
    manager = env.termination_manager
    if env.num_envs != 1:
        raise ValueError("Force audit capture requires exactly one environment")
    if getattr(manager, "_knife_force_audit_capture_installed", False):
        raise ValueError("Force audit capture is already installed")
    original = manager.compute
    terms = [(name, manager.get_term_cfg(name)) for name in manager.active_terms]
    command = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    array = lambda x: x.detach().cpu().tolist()
    current = {}
    manager._knife_force_audit_capture_installed = True
    for name, cfg in terms:
        def capture_term(*args, _name=name, _original=cfg.func, **kwargs):
            value = _original(*args, **kwargs)
            if _name in current or tuple(value.shape) != (1,):
                raise ValueError("Expected exactly one call per term and one environment")
            current[_name] = bool(value[0])
            return value
        cfg.func = capture_term

    def compute():
        current.clear()
        result = original()
        if set(current) != {name for name, _ in terms}:
            raise ValueError("Missing current termination results")
        causes = dict(current)
        computed_done = bool(result[0])
        manager_terminated = bool(manager.terminated[0])
        manager_truncated = bool(manager.time_outs[0])
        if any(causes.values()) != computed_done or (manager_terminated or manager_truncated) != computed_done:
            raise ValueError("Current termination union differs from compute result")
        step = int(command.timesteps_in_current_demo[0])
        pads, raw = {}, False
        for name in SENSORS:
            matrix = env.scene.sensors[name].data.force_matrix_w
            if matrix is None or tuple(matrix.shape) != (1, 1, 1, 3) or torch.isinf(matrix).any():
                raise ValueError(f"Missing/malformed filtered force_matrix_w: {name}")
            vector = torch.nan_to_num(matrix[0, 0, 0], nan=0.0)
            magnitude = float(torch.linalg.vector_norm(vector))
            raw |= magnitude > 10.
            pads[name] = {"normal_force_vector_w_N": array(vector), "normal_magnitude_N": magnitude,
                          "nan_as_zero_components": int(torch.isnan(matrix).sum())}
        allowed = old_allowed(step, env.step_dt)
        root = env.action_manager.get_term("root_pose")
        values = (robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel,
                  robot.data.applied_torque, env.scene["object"].data.root_state_w)
        if any(not torch.isfinite(v).all() for v in values):
            raise ValueError("Nonfinite pre-reset state/effort estimate")
        row = {"control": len(rows) + 1, "reference_timestep": step, "reference_phase_s": step * env.step_dt,
               "source_frame_30fps": 132 + step, "source_native_frame_120fps": 528 + 4 * step,
               "reference_phase_fraction": step / 429, "episode_control": int(env.episode_length_buf[0]),
               "pads": pads, "old_allowed": allowed, "raw_over_10N": raw,
               "raw_over_10N_outside": raw and not allowed, "causes": causes,
               "computed_done": computed_done, "manager_terminated": manager_terminated,
               "manager_truncated": manager_truncated,
               "hit_table_consistent": causes["hit_table"] == (raw and not allowed),
               "root_position_error_m": array(root.target_pos - (robot.data.root_pos_w - env.scene.env_origins))[0],
               "root_rotation_error_rad": float(quat_error_magnitude(root.target_quat, robot.data.root_quat_w)[0]),
               "finger_position_error_rad": array(robot.data.joint_pos_target - robot.data.joint_pos)[0],
               "finger_effort_estimate_Nm": array(robot.data.applied_torque)[0],
               "object_position_drift_m": array(command.current_object_pos - command.target_object_pos)[0],
               "object_rotation_drift_rad": float(quat_error_magnitude(command.current_object_quat, command.target_object_quat)[0])}
        rows.append(row)
        if not row["hit_table_consistent"]:
            raise ValueError("Raw filtered force/window vs actual hit_table mismatch")
        return result

    manager.compute = compute


def record_step_result(report, terminated, truncated):
    """Check returned flags against the pre-reset capture, without changing manager buffers."""
    if tuple(terminated.shape) != (1,) or tuple(truncated.shape) != (1,):
        raise ValueError("Expected single-environment returned termination flags")
    row = report["controls"][-1]
    returned_terminated, returned_truncated = bool(terminated[0]), bool(truncated[0])
    returned_done = returned_terminated or returned_truncated
    row.update(returned_terminated=returned_terminated, returned_truncated=returned_truncated,
               returned_done=returned_done)
    if (returned_done != row["computed_done"] or returned_terminated != row["manager_terminated"]
            or returned_truncated != row["manager_truncated"]):
        raise ValueError("Returned termination flags differ from pre-reset compute")
    report["reset_count"] += int(returned_done)
    if report["reset_count"] != sum(int(r["computed_done"]) for r in report["controls"]):
        raise ValueError("Reset count differs from captured current termination union")


class Video:
    def __init__(self, path):
        self.path, self.writer, self.frames = Path(path), None, 0
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self.identity = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        os.close(fd)

    def start(self, sim):
        import imageio.v2 as imageio
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        from PIL import ImageFont
        stat = self.path.lstat()
        if self.path.is_symlink() or (stat.st_dev, stat.st_ino) != self.identity or stat.st_size:
            raise ValueError("Video reservation changed")
        camera_path = "/World/KnifeForceAuditCamera"
        camera = UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), camera_path)
        camera.CreateProjectionAttr().Set("orthographic")
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(.01, 100.))
        camera.CreateHorizontalApertureAttr().Set(14.)
        camera.CreateVerticalApertureAttr().Set(14. * 720 / 1280)
        sim.set_camera_view((.45, -.8, .75), (-.25, 0., .12), camera_prim_path=camera_path)
        self.product = rep.create.render_product(camera_path, (1280, 720))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annotator.attach([self.product])
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        sim.forward()
        for _ in range(4):
            sim.render()
        self.writer = imageio.get_writer(str(self.path), fps=30, codec="libx264", macro_block_size=16)

    def capture(self, sim, resets, row):
        import numpy as np
        from PIL import Image, ImageDraw
        sim.forward()
        sim.render()
        rgb = np.asarray(self.annotator.get_data())
        if rgb.shape not in ((720, 1280, 3), (720, 1280, 4)) or rgb.dtype != np.uint8 or not rgb[:, :, :3].any():
            raise ValueError("Invalid RGB frame")
        image = Image.fromarray(rgb[:, :, :3])
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1279, 90), fill="black")
        draw.text((14, 5), "Knife | ZERO ACTIONS | NOT POLICY", font=self.font, fill="white")
        draw.text((14, 48), f"Control {self.frames + 1}/240 | resets {resets} | pre-reset phase {row['reference_phase_s']:.3f}s",
                  font=self.font, fill="white")
        self.writer.append_data(np.asarray(image))
        self.frames += 1

    def close(self):
        if self.writer is not None:
            writer, self.writer = self.writer, None
            writer.close()


def runtime(directory, video, report, app):
    import torch
    import omni.usd
    from pxr import Usd, UsdPhysics
    from isaaclab.envs import ManagerBasedRLEnv
    os.environ["HAND2_USD_CACHE_DIR"] = str(directory / "hand_usd")
    os.environ["HAND2_NUM_ENVS"] = "1"
    os.environ.setdefault("REGRIND_DATA_DIR", str(REPO / "data"))
    cfg = make_config(directory)
    env = ManagerBasedRLEnv(cfg=cfg)  # No gym.make, no new registration, no policy.
    if env.num_envs != 1 or env.action_manager.total_action_dim != 26:
        raise ValueError("Expected exactly one environment and 26 actions")
    obj = env.scene["object"]
    mass = obj.root_physx_view.get_masses()
    if not torch.isfinite(mass).all() or not (mass > 0).all():
        raise ValueError("Invalid actual object mass")
    report["actual_object_mass_kg"] = mass.cpu().tolist()
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0/Object")
    colliders = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    report["actual_object_colliders"] = colliders
    if len(colliders) != 6:
        raise ValueError(f"Expected six actual object colliders, got {len(colliders)}")
    env.reset()  # Only explicit reset; ordinary task resets stay enabled.
    motion = env.command_manager.get_term("motion")
    if int(motion.timesteps_in_current_demo[0]) != 0 or int(motion.current_demo_length[0]) != 429:
        raise ValueError("Native full crop must start at frame zero, 429 controls")
    robot = env.scene["robot"]
    report["joint_names"] = list(robot.joint_names)
    report["finger_effort_note"] = "applied_torque is a clipped PD estimate, NOT measured solver joint torque"
    report["actual_hand_body_mass_kg"] = robot.root_physx_view.get_masses().cpu().tolist()
    report["actual_disable_gravity"] = {}
    for name, asset in (("hand", robot), ("object", obj)):
        disabled = asset.root_physx_view.get_disable_gravities()
        report["actual_disable_gravity"][name] = disabled.cpu().tolist()
        if torch.any(disabled != 0):
            raise ValueError(f"Actual {name} gravity must remain ON")
    report["actual_scene_gravity"] = list(env.sim.physics_sim_view.get_gravity())
    if tuple(report["actual_scene_gravity"]) != (0., 0., -9.81):
        # PhysX may expose float32 gravity; tolerate representation only.
        if not torch.allclose(torch.tensor(report["actual_scene_gravity"]), torch.tensor([0., 0., -9.81]), rtol=0., atol=1e-6):
            raise ValueError("Actual scene gravity changed")
    install_capture(env, report["controls"])
    video.start(env.sim)
    actions = torch.zeros((1, 26), device=env.device)
    with torch.inference_mode():
        for _ in range(CONTROLS):
            if not app.is_running() or not env.sim.is_playing():
                raise RuntimeError("Simulation stopped/paused before 240 controls")
            _, reward, terminated, truncated, _ = env.step(actions)
            report["successful_controls"] += 1
            if len(report["controls"]) != report["successful_controls"] or not torch.isfinite(reward).all():
                raise ValueError("Missing pre-reset control or nonfinite reward")
            record_step_result(report, terminated, truncated)
            video.capture(env.sim, report["reset_count"], report["controls"][-1])
    # Do not env.close() here: SimulationContext close may destroy the process.


def validate_outputs(video, record):
    video, record = Path(video).absolute(), Path(record).absolute()
    if video != VIDEO or record != RECORD or video.parent.resolve() != ARTIFACTS.resolve():
        raise ValueError("Only the explicitly scoped video and existing run.json are allowed")
    if video.exists() or video.is_symlink() or record.is_symlink() or not record.is_file():
        raise ValueError("Require a fresh video and an existing regular record")
    if KEY in json.loads(record.read_text()):
        raise ValueError("Audit record key already exists")
    return video, record


def flush_record(path, report):
    """Read the latest JSON under lock, preserve every other key, fsync before app.close."""
    with Path(path).open("r+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        current = json.load(stream)
        if KEY in current:
            raise ValueError("Refusing to overwrite existing audit key")
        current[KEY] = report
        payload = json.dumps(current, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        stream.seek(0)
        stream.write(payload)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-video")
    parser.add_argument("--record")
    args = parser.parse_args(argv)
    if args.run and (not args.output_video or not args.record):
        parser.error("--run requires --output-video and --record")
    if not args.run and (args.output_video or args.record):
        parser.error("Read-only checks take no output arguments")
    checked = precheck()
    if not args.run:
        print(json.dumps(checked, indent=2))
        return 0
    output, record = validate_outputs(args.output_video, args.record)
    report = {**checked, "status": "failed", "approved_for_training": False, "policy": "NONE",
              "scope": "LOCAL 240 successful zero-action controls, NOT full-trajectory or training approval",
              "allowed_windows_crop_30fps_closed": WINDOWS, "force_threshold_N": 10.,
              "successful_controls": 0, "reset_count": 0, "controls": [], "video": str(output),
              "root_profile": "com_bounded_v3", "root_gains": [300., 30., 3., .02],
              "finger_action_scale": .32, "table_top_m": .002,
              "force_source": "Isaac filtered pad-to-table normal force_matrix_w; no Drake force inference",
              "video_sampling": "30 fps, 1280x720, one post-step/post-auto-reset image per successful control",
              "video_camera_near_m": .01, "initial_reset_count": 1,
              "mask_semantics": "native per-control closed windows; no persistence counter"}
    app, video, temporary = None, None, None
    try:
        video = Video(output)
        temporary = tempfile.TemporaryDirectory(prefix="hand2-knife-force-")
        directory = Path(temporary.name)
        report["importer_alias_transformation"] = stage_inputs(directory, checked)
        verify_pins(checked["source_hashes_before"])
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
        runtime(directory, video, report, app)
        report["status"] = "completed_not_training_approved"
    except BaseException:
        report["error"] = traceback.format_exc()
    # All durable recording and owned-cache cleanup precede destructive app.close.
    try:
        if video is not None:
            video.close()
            report["video_frames"] = video.frames
            report["video_writer_closed"] = True
        report["source_hashes_after"] = verify_pins(checked["source_hashes_before"])
    except BaseException:
        report["status"] = "failed"
        report["finalization_error"] = traceback.format_exc()
    try:
        if temporary is not None:
            temporary.cleanup()
            report["temporary_cache_removed"] = True
    except BaseException:
        report["status"] = "failed"
        report["cleanup_error"] = traceback.format_exc()
    report["raw_over_10N_outside_controls"] = [r["control"] for r in report["controls"] if r["raw_over_10N_outside"]]
    flush_record(record, report)
    print(json.dumps({"status": report["status"], "successful_controls": report["successful_controls"]}), flush=True)
    code = 0 if report["status"] == "completed_not_training_approved" else 1
    if app is not None:
        app.close(skip_cleanup=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
