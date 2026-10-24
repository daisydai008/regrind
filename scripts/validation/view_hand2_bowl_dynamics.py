#!/usr/bin/env python3
"""Interactive actual Hand2 Bowl task, zero residual actions, NO POLICY.

Uncalibrated virtual root PD, NOT an official arm controller; generic vendor
fingers unchanged. Optional hand gravity uses root-only ideal net weight support
(not arm dynamics). Bowl gravity, table and task terminations are unchanged. Bowl moves by dynamics except normal episode resets.
Run in the existing Isaac environment with a NEW caller-owned --output-dir.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

TASK = "Regrind-WujiHand2-Bowl-S-Play-v0"
PROFILE = "inertia-derived"
REPO = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DynamicsGuard(RuntimeError):
    """Latch a failure before the original root action can write a wrench."""


def configure_root(cfg, probe):
    root = cfg.actions.root_pose
    if (root.kp_pos, root.kd_pos, root.kp_rot, root.kd_rot) != (300.0, 30.0, 3.0, 0.3):
        raise ValueError("Unexpected native root gains")
    root.kp_rot, root.kd_rot = probe.ROOT_PROFILES[PROFILE]


def configure_gravity(cfg, hand_gravity):
    if hand_gravity not in ("off", "on"):
        raise ValueError("Expected hand gravity off/on")
    if cfg.scene.robot.spawn.rigid_props.disable_gravity is not True:
        raise ValueError("Expected native hand gravity OFF")
    if hand_gravity == "on":
        if tuple(cfg.sim.gravity) != (0.0, 0.0, -9.81):
            raise ValueError("Gravity ON requires scene gravity (0, 0, -9.81)")
        cfg.scene.robot.spawn.rigid_props.disable_gravity = False


def zero_target_guard(robot, array):
    import numpy as np
    for name in ("joint_vel_target", "joint_effort_target"):
        values = array(getattr(robot.data, name))
        if values.shape != (1, 20) or not np.isfinite(values).all() or np.any(values != 0):
            raise ValueError(f"Expected exactly zero finite {name} for all 20 joints")
    return {"qdot_target_exactly_zero": True, "effort_target_exactly_zero": True}


def audit_fingers(robot, auditor, array):
    import numpy as np
    ids = auditor.joint_indices(robot)
    applied = auditor.check_applied_parameters(robot, ids)
    actuator = robot.actuators["fingers"]
    if actuator.is_implicit_model is not True:
        raise ValueError("fingers must be an actual implicit actuator")
    indices = actuator.joint_indices
    indices = list(range(20))[indices] if isinstance(indices, slice) else list(array(indices) if hasattr(indices, "detach") else indices)
    if (len(indices) != 20 or set(indices) != set(range(20))
            or list(actuator.joint_names) != [robot.joint_names[i] for i in indices]):
        raise ValueError("Non-bijective fingers actuator joint_indices/names")
    order = [indices.index(i) for i in ids]
    actual = {}
    for field, key in (("stiffness", "kp"), ("damping", "kv"),
                       ("effort_limit", "effort"), ("effort_limit_sim", "effort")):
        values = array(getattr(actuator, field))
        if (values.shape != (1, 20) or not np.isfinite(values).all()
                or not np.allclose(values[0, order], applied[key], rtol=2e-6, atol=1e-8)):
            raise ValueError(f"Actual fingers actuator mismatch: {field}")
        actual[field] = values[0, order].tolist()
    names = auditor.contract.JOINT_NAMES
    return {"joint_indices": dict(zip(names, map(int, ids))),
            "applied": {n: {k: v[i] for k, v in applied.items()} for i, n in enumerate(names)},
            "actuator": {"name": "fingers", "is_implicit_model": True,
                         "values": {n: {k: v[i] for k, v in actual.items()} for i, n in enumerate(names)}},
            "targets": zero_target_guard(robot, array)}


def guarded_pd(term, probe, array):
    """Probe uses WORLD targets; action targets are relative to env origin."""
    import numpy as np
    state = array(term._asset.data.root_state_w)
    origin = array(term._env.scene.env_origins)
    pos, quat = array(term.target_pos), array(term.target_quat)
    if state.shape != (1, 13) or origin.shape != (1, 3) or pos.shape != (1, 3) or quat.shape != (1, 4):
        raise ValueError("Root guard requires exactly one environment")
    target = np.r_[pos[0] + origin[0], quat[0]]
    return probe.pd_wrench_guard(probe.root_pd_wrench(target, state[0], PROFILE))


def native_pd(term):
    """Exact native float32 formula, including its quaternion hemisphere choice."""
    import torch
    from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat
    state = term._asset.data.root_state_w
    pos = term._asset.data.root_pos_w - term._env.scene.env_origins
    error = quat_mul(term.target_quat, quat_conjugate(term._asset.data.root_quat_w))
    error = torch.where(error[..., 0:1] < 0, -error, error)
    return (term.kp_pos * (term.target_pos - pos) - term.kd_pos * state[:, 7:10],
            term.kp_rot * axis_angle_from_quat(error) - term.kd_rot * state[:, 10:13])


def install_root_guard(term, probe, array, native=native_pd, *, hand_gravity="off",
                       tensor=None, initial_masses=None):
    original = term.apply_actions
    robot = term._asset
    if hand_gravity not in ("off", "on"):
        raise ValueError("Expected hand gravity off/on")
    if hand_gravity == "on":
        if tensor is None or initial_masses is None or len(robot.body_names) != 33:
            raise ValueError("Gravity support requires audited 33-body masses and tensor adapter")
        root_id = robot.body_names.index("r_mount")
        initial_masses = probe.finite_array(initial_masses, (33,)).copy()

    def apply_guarded(self):
        try:
            expected = probe.root_controller(PROFILE)
            if any(getattr(self, key) != expected[key] for key in ("kp_pos", "kd_pos", "kp_rot", "kd_rot")):
                raise ValueError("Runtime root gains differ from guarded profile")
            zero_target_guard(robot, array)
            guarded_pd(self, probe, array)
            # Check exact native tensor arithmetic too; no clipping/replacement.
            pd = probe.pd_wrench_guard(tuple(array(value)[0] for value in native(self)))
            if hand_gravity == "on":
                import numpy as np
                masses = probe.finite_array(array(robot.root_physx_view.get_masses()), (1, 33))[0]
                if np.any(masses <= 0) or not np.array_equal(masses, initial_masses):
                    raise ValueError("Live hand masses differ from initial audited contract")
                root = probe.finite_array(array(robot.data.root_pos_w), (1, 3))[0]
                coms = probe.finite_array(array(robot.data.body_com_pos_w), (1, 33, 3))[0]
                support = probe.root_support_wrench(masses, coms, root, probe.GRAVITY)
                return probe.apply_root_wrench(robot, root_id, pd, support, root, tensor)
        except Exception as error:
            raise DynamicsGuard(str(error)) from error
        # Preserve native positions=None: wrench at root LINK origin, not COM.
        return original()

    term.apply_actions = MethodType(apply_guarded, term)


def check_finite(value, torch):
    if isinstance(value, dict):
        for child in value.values():
            check_finite(child, torch)
    elif isinstance(value, (tuple, list)):
        for child in value:
            check_finite(child, torch)
    elif torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
        raise DynamicsGuard("Nonfinite observation/state")


def live_gravity_flags(asset, names, disable, array):
    import numpy as np
    values = array(asset.root_physx_view.get_disable_gravities())
    if values.shape != (1, len(names)) or not np.isfinite(values).all() or not np.all(values == int(disable)):
        raise ValueError("Unexpected live PhysX disable-gravity flags")
    return dict(zip(names, values[0].astype(int).tolist()))


def audit_runtime(env, factory, hand_gravity="off"):
    """Read actual PhysX and USD flags; fail closed before any env.step."""
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdPhysics, PhysxSchema
    array = lambda value: value.detach().cpu().numpy()
    robot = env.scene["robot"]
    mass = factory.check_runtime_masses(robot)
    stage = omni.usd.get_context().get_stage()
    gravity_report = {}
    for name, asset, disable, count in (("Robot", robot, hand_gravity == "off", 33),
                                        ("Object", env.scene["object"], False, 1)):
        names = list(asset.body_names)
        if len(names) != count or len(set(names)) != count:
            raise ValueError(f"Unexpected {name} live body names")
        live = live_gravity_flags(asset, names, disable, array)
        root = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        if not root.IsValid():
            raise ValueError(f"Missing {name} prim")
        bodies = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != count or {p.GetName() for p in bodies} != set(names):
            raise ValueError(f"Unexpected {name} USD rigid-body names/count")
        usd = {}
        for prim in bodies:
            gravity = PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()
            body = UsdPhysics.RigidBodyAPI(prim)
            if gravity is not disable or body.GetKinematicEnabledAttr().Get() or not body.GetRigidBodyEnabledAttr().Get():
                raise ValueError(f"Unexpected gravity/kinematic/body-enabled flag: {prim.GetPath()}")
            usd[prim.GetName()] = gravity
        gravity_report[name] = {"physx_disable_gravity": live, "usd_disable_gravity": usd}
    object_mass = array(env.scene["object"].root_physx_view.get_masses())
    if object_mass.size != 1 or not np.isfinite(object_mass).all() or not (object_mass > 0).all():
        raise ValueError("Invalid live Bowl mass")
    return {"hand_total_mass_kg": mass["total_mass_kg_per_env"][0],
            "hand_body_mass_kg": mass["body_mass_kg_per_env"][0],
            "bowl_mass_kg": float(object_mass.reshape(-1)[0]), "gravity": gravity_report}


def save_parameter_audit(output, report):
    # Exclusive, single durable write, before the first env.step.
    with (output / "parameter_audit.json").open("x") as stream:
        json.dump(report, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def ui_update_only(app, sim):
    # IsaacLab render() uses this setting too: update UI without a physics tick,
    # even if the user presses Play during a latched failure.
    sim.set_setting("/app/player/playSimulations", False)
    try:
        app.update()
    finally:
        sim.set_setting("/app/player/playSimulations", True)


def recording_frame_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1800:
        raise ValueError("--record-frames must be an integer in [1, 1800]")
    return value


class VideoRecording:
    """Own one exclusive MP4 reservation and stream one frame per successful step."""
    resolution = (1280, 720)
    camera_target = (-0.48731698, 0.05815392, 0.21418751)
    camera_offset = (0.40, -0.65, 0.45)
    horizontal_width = 1.40

    def __init__(self, path, frames=240):
        self.limit = recording_frame_count(frames)
        # Inspect the unnormalized path too, so symlink/../ cannot bypass rejection.
        raw = Path(path).expanduser().absolute()
        if any(p.is_symlink() for p in (raw, *raw.parents)):
            raise ValueError(f"Symlink video path rejected: {raw}")
        self.path = Path(os.path.abspath(raw))
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("--record-video must name a new .mp4 file")
        if not self.path.parent.is_dir():
            raise ValueError("Video parent directory must already exist")
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            stat = os.fstat(fd)
            self.identity = (stat.st_dev, stat.st_ino)
        finally:
            os.close(fd)
        self.writer = None
        self.started = False
        self.frames = 0

    def start(self, sim):
        import imageio.v2 as imageio
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom
        from PIL import ImageFont

        stat = self.path.lstat()
        if self.path.is_symlink() or (stat.st_dev, stat.st_ino) != self.identity or stat.st_size:
            raise ValueError("Video reservation changed before writer startup")
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.heading_font = ImageFont.truetype(font, 32)
        self.status_font = ImageFont.truetype(font, 26)
        camera_path = "/World/Hand2RecordingCamera"
        camera = UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), camera_path)
        camera.CreateProjectionAttr().Set("orthographic")
        camera.CreateClippingRangeAttr().Set((0.01, 100.0))  # USD default near=1 m clips this close-up.
        # Full pinned H5 root/object/all named keypoints: projected spans
        # 0.49793684 x 0.43874866 m. 0.6 m width clips vertical motion.
        # Bowl visual mesh over all H5 poses reaches camera y=+0.27452614 m;
        # 1.40 m extent keeps it below the top bar (+0.301875 m), with margin.
        camera.CreateHorizontalApertureAttr().Set(self.horizontal_width * 10)
        camera.CreateVerticalApertureAttr().Set(self.horizontal_width * 10 * 720 / 1280)
        sim.set_camera_view(tuple(a + b for a, b in zip(self.camera_target, self.camera_offset)),
                            self.camera_target, camera_prim_path=camera_path)
        self.product = rep.create.render_product(camera_path, self.resolution)
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annotator.attach([self.product])
        sim.forward()
        for _ in range(4):
            sim.render()  # Render-only warmup: no physics or env.reset.
        self.writer = imageio.get_writer(str(self.path), fps=30, codec="libx264", macro_block_size=16)
        self.started = True

    def audit(self):
        return {"path": str(self.path), "requested_frames": self.limit, "fps": 30,
                "resolution": list(self.resolution), "capture": "one frame per successful env.step; no wall-clock sampling",
                "camera_target": list(self.camera_target), "camera_eye_offset": list(self.camera_offset),
                "orthographic_horizontal_width_m": self.horizontal_width,
                "reference_projected_span_m": [0.49793684, 0.43874866],
                "camera_bounds_source": "pinned H5 all 1985 frames: robot_pos, object_pos, all 21 named robot_keypoints",
                "bowl_mesh_projected_y_m": [-0.26647712, 0.27452614],
                "geometry_margin_scope": "Bowl visual mesh transformed at all H5 poses; hand keypoints plus framing allowance, not a hand mesh audit",
                "approval": "NOT a dynamics/policy approval; completion requires successful close and frame count"}

    def capture(self, sim, steps, resets, phase_elapsed, hand_gravity):
        import numpy as np
        from PIL import Image, ImageDraw
        if steps != self.frames + 1 or steps > self.limit or not math.isfinite(phase_elapsed) or phase_elapsed < 0:
            raise ValueError("Invalid recording step/phase counter")
        sim.forward()  # Synchronize post-step/post-auto-reset transforms; no physics.
        sim.render()
        rgb = np.asarray(self.annotator.get_data())
        if rgb.shape not in ((720, 1280, 3), (720, 1280, 4)) or rgb.dtype != np.uint8 or not np.any(rgb[:, :, :3]):
            raise ValueError(f"Invalid/empty RGB render: {rgb.shape}")
        image = Image.fromarray(rgb[:, :, :3])
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1279, 83), fill="black")
        draw.rectangle((0, 676, 1279, 719), fill="black")
        draw.text((16, 5), "Hand 2.2 RIGHT | Isaac dynamics (NO POLICY)", font=self.heading_font, fill="white")
        draw.text((16, 46), f"Finger gravity {hand_gravity.upper()} | Bowl: contact + gravity | 30 Hz",
                  font=self.status_font, fill="white")
        draw.text((16, 682), f"Frame {steps}/{self.limit} | episode_resets {resets} (+1 initial) | sim {steps/30:.2f}s | phase {phase_elapsed:.2f}s",
                  font=self.status_font, fill="white")
        self.writer.append_data(np.asarray(image))  # Stream, never accumulate frames.
        self.frames += 1

    def close(self):
        if self.writer is not None:
            writer, self.writer = self.writer, None
            writer.close()
        if not self.started:
            # Remove only our still-empty reservation, never a pre-existing/replaced file.
            try:
                stat = self.path.lstat()
            except FileNotFoundError:
                return
            if not self.path.is_symlink() and (stat.st_dev, stat.st_ino) == self.identity and stat.st_size == 0:
                self.path.unlink()


def run(app, probe, factory, max_seconds, output, hand_gravity="off", *, recording=None):
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import regrind.tasks  # noqa: F401

    gravity_label = "ON + root-only ideal net weight support (NOT arm dynamics)" if hand_gravity == "on" else "OFF"
    finger_status, status, motion_status = (SimpleNamespace(text="") for _ in range(3))
    if recording is None:
        import omni.ui as ui
        window = ui.Window("DYNAMIC BOWL — NO POLICY", width=750, height=260)
        with window.frame:
            with ui.VStack(spacing=3):
                ui.Label("Bowl: gravity + contact only; normal episode resets; NO per-frame Bowl writes")
                ui.Label(f"Hand gravity {gravity_label} | Bowl gravity ON | table top 0.012 m; known initial overlap", word_wrap=True)
                ui.Label("Virtual root PD UNCALIBRATED, NOT official arm controller | generic vendor fingers | NO PPO", word_wrap=True)
                finger_status = ui.Label("Official generic finger contract: audit pending | NO PPO", word_wrap=True)
                status = ui.Label("STARTING: runtime mass/gravity checks pending", word_wrap=True)
                motion_status = ui.Label("Object actual/reference position error: pending", word_wrap=True)
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1, use_fabric=True)
    configure_root(cfg, probe)
    configure_gravity(cfg, hand_gravity)
    if cfg.scene.num_envs != 1 or cfg.decimation != 4 or not math.isclose(cfg.sim.dt, 1 / 120):
        raise ValueError("Expected one environment, 120 Hz physics / 30 Hz actions")
    if not math.isclose(cfg.scene.table.init_state.pos[2] + cfg.scene.table.spawn.size[2] / 2, 0.012):
        raise ValueError("Unexpected table height")
    wrapped = gym.make(TASK, cfg=cfg)
    env = wrapped.unwrapped
    sim = env.sim
    array = lambda value: value.detach().cpu().numpy()
    # gym.make initializes PhysX; audit BEFORE gym.reset or any env.step.
    report = audit_runtime(env, factory, hand_gravity)
    auditor = load_module("hand2_live_auditor", Path(__file__).with_name("validate_hand2_right_isaac.py"))
    robot = env.scene["robot"]
    report["fingers"] = audit_fingers(robot, auditor, array)
    term = env.action_manager.get_term("root_pose")
    tensor = lambda value: torch.as_tensor(value, device=env.device, dtype=torch.float32)
    install_root_guard(term, probe, array, hand_gravity=hand_gravity, tensor=tensor,
                       initial_masses=[report["hand_body_mass_kg"][n] for n in robot.body_names])
    if env.num_envs != 1 or env.action_manager.total_action_dim != 26:
        raise ValueError("Expected single-environment 26-D action manager")
    obs, _ = wrapped.reset()  # The only explicit reset; task owns episode auto-resets.
    report["fingers"]["post_reset_targets"] = zero_target_guard(robot, array)
    report.update({"provenance": probe.preflight(factory.contract),
                   "official_parameter_status": "verified generic vendor nominal finger contract; NOT reorient; NOT hardware calibration",
                   "parameter_status": factory.contract.PARAMETER_STATUS,
                   "virtual_root_controller": {**probe.root_controller(PROFILE), "official_arm_controller": False},
                   "hand_gravity": hand_gravity, "scene_gravity": list(cfg.sim.gravity),
                   "support": "root-only ideal net weight support, excludes Bowl; NOT arm dynamics; no joint feedforward" if hand_gravity == "on" else "none",
                   "physics_dt_s": cfg.sim.dt, "physics_hz": 120, "control_hz": 30,
                   "finger_torque_disclaimer": "robot.data.applied_torque is a clipped PD ESTIMATE, NOT the implicit PhysX solver law or measured joint torque",
                   "policy": "NONE; NO PPO", "known_initial_table_overlap_unchanged": True})
    if recording is not None:
        report["recording"] = recording.audit()
    save_parameter_audit(output, report)
    actions = torch.zeros((1, 26), device=env.device)
    command = env.command_manager.get_term("motion")
    steps, resets, frozen = 0, 0, None
    start = last_print = time.monotonic()
    print(f"READY task={TASK} NO_POLICY kp_pos=300 kd_pos=30 kp_rot={term.kp_rot} kd_rot={term.kd_rot} "
          f"hand_mass_kg={report['hand_total_mass_kg']} bowl_mass_kg={report['bowl_mass_kg']} hand_gravity={gravity_label} bowl_gravity=ON "
          "official_finger_contract=VERIFIED generic_vendor=true NO_PPO NOT_official_arm_controller "
          "table_top_m=0.012 UNCALIBRATED known_table_overlap=true no_per_frame_Bowl_writes=true", flush=True)
    finger_status.text = "Official generic finger contract VERIFIED (uncalibrated vendor parameters) | NO PPO"
    if recording is not None:
        recording.start(sim)
    while app.is_running() and (steps < recording.limit if recording is not None else time.monotonic() - start < max_seconds):
        tick = time.monotonic()
        if recording is not None and not sim.is_playing():
            frozen = "Recording simulation stopped/paused"
            break
        if frozen is None and sim.is_playing():
            try:
                with torch.inference_mode():
                    check_finite((obs, env.scene["robot"].data.root_state_w,
                                  env.scene["robot"].data.joint_pos, env.scene["robot"].data.joint_vel,
                                  env.scene["object"].data.root_state_w), torch)
                    obs, reward, terminated, truncated, _ = wrapped.step(actions)
                    steps += 1
                    resets += int(bool((terminated | truncated).any()))
                    check_finite((obs, reward, env.scene["robot"].data.root_state_w,
                                  env.scene["robot"].data.joint_pos, env.scene["robot"].data.joint_vel,
                                  env.scene["object"].data.root_state_w), torch)
                    if steps % 30 == 0 or steps == 1:
                        actual = array(command.current_object_pos)[0]
                        reference = array(command.target_object_pos)[0]
                        probe.finite_array(actual, (3,))
                        probe.finite_array(reference, (3,))
                        error = np.linalg.norm(actual - reference)
                        motion_status.text = (f"Object actual {np.round(actual, 4)} m | reference {np.round(reference, 4)} m "
                                              f"| error {error:.4f} m (post-step/post-reset)")
                    if recording is not None:
                        phase_elapsed = float(array(command.timesteps_in_current_demo)[0]) / 30
                        recording.capture(sim, steps, resets, phase_elapsed, hand_gravity)
            except Exception as error:
                frozen = f"{type(error).__name__}: {error}"
                sim.pause()
                print(f"FROZEN — NOT SUCCESS: {frozen}", flush=True)
                if recording is not None:
                    break  # Headless failures stop immediately, never wait for GUI timeout.
        if frozen is not None:
            sim.pause()
        status.text = (f"{'FROZEN — NOT SUCCESS: ' + frozen if frozen else ('RUNNING' if sim.is_playing() else 'PAUSED')} "
                       f"| steps={steps} episode_resets={resets} (+1 initial) | wall={time.monotonic()-start:.1f}s sim={steps/30:.1f}s")
        if frozen is not None or not sim.is_playing():
            ui_update_only(app, sim)
        if time.monotonic() - last_print >= 1:
            print(f"steps={steps} episode_resets={resets} wall_s={time.monotonic()-start:.1f} "
                  f"no_per_frame_Bowl_writes=true state={'FROZEN' if frozen else 'live/paused'}", flush=True)
            last_print = time.monotonic()
        if recording is None:
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - tick)))
    if recording is not None and (frozen is not None or recording.frames != recording.limit):
        print(f"RECORDING_FAILED frames={recording.frames}/{recording.limit} reason={frozen or 'application stopped'} path={recording.path}", flush=True)
        return 1
    return 1 if frozen is not None else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW local directory; parent must exist; USD caches and parameter_audit.json")
    parser.add_argument("--max-seconds", type=float, default=3600, help="Wall-clock GUI limit (default: 3600)")
    parser.add_argument("--hand-gravity", choices=("off", "on"), default="off", help="Default off; on adds root-only ideal net weight support, NOT arm dynamics")
    parser.add_argument("--record-video", type=Path, help="NEW .mp4; existing parent required; enables headless cameras")
    parser.add_argument("--record-frames", type=int, default=240, help="Successful steps/frames at 30 Hz (1..1800, default: 240 = 8 sim seconds)")
    args = parser.parse_args(argv)  # --help imports no numpy/torch/Isaac.
    try:
        recording_frame_count(args.record_frames)
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    app = recording = None
    result = 1
    try:
        # Exclusive reservation precedes app startup, output directory and USD caches.
        if args.record_video is not None:
            recording = VideoRecording(args.record_video, args.record_frames)
        probe = load_module("hand2_live_probe", Path(__file__).with_name("validate_hand2_free_space.py"))
        factory = load_module("hand2_live_factory", probe.ROBOTS / "free_hand2_right_hand.py")
        probe.preflight(factory.contract)
        data = Path(os.environ.setdefault("REGRIND_DATA_DIR", str(REPO / "data")))
        if not data.is_dir() or factory.contract.safe_path(data / "hand2_right/bowl/retargeted_120fps.h5") != probe.REFERENCE:
            raise ValueError("REGRIND_DATA_DIR must resolve to the preflight-pinned task data")
        output = factory.contract.new_path(args.output_dir)
        output.mkdir()
        os.environ["HAND2_USD_CACHE_DIR"] = str(output / "hand_usd_cache")
        from isaaclab.app import AppLauncher
        if recording is None:
            app = AppLauncher(headless=False, device="cuda:0").app
            result = run(app, probe, factory, args.max_seconds, output, args.hand_gravity)
        else:
            app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
            result = run(app, probe, factory, args.max_seconds, output, args.hand_gravity, recording=recording)
    except Exception as error:
        if args.record_video is None:
            raise
        print(f"RECORDING_FAILED {type(error).__name__}: {error} path={args.record_video}", flush=True)
        result = 1
    finally:
        try:
            if recording is not None:
                recording.close()  # Always finalize partial videos; never delete recorded data.
        except Exception as error:
            print(f"RECORDING_FAILED writer close: {type(error).__name__}: {error}", flush=True)
            result = 1
        finally:
            # Immediate Isaac close can terminate the process instead of returning.
            # Publish completion only after writer.close, but before application close.
            if recording is not None and result == 0:
                if recording.frames != recording.limit:
                    print(f"RECORDING_FAILED frame count={recording.frames}/{recording.limit}", flush=True)
                    result = 1
                else:
                    print(f"RECORDING_COMPLETE frames={recording.frames} path={recording.path}", flush=True)
            if app is not None:
                app.close(skip_cleanup=True)  # Only the app created by this invocation.
    return result


if __name__ == "__main__":
    raise SystemExit(main())
