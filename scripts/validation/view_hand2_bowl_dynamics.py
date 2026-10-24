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
from types import MethodType

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


def run(app, probe, factory, max_seconds, output, hand_gravity="off"):
    import gymnasium as gym
    import numpy as np
    import torch
    import omni.ui as ui
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import regrind.tasks  # noqa: F401

    gravity_label = "ON + root-only ideal net weight support (NOT arm dynamics)" if hand_gravity == "on" else "OFF"
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
    while app.is_running() and time.monotonic() - start < max_seconds:
        tick = time.monotonic()
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
            except Exception as error:
                frozen = f"{type(error).__name__}: {error}"
                sim.pause()
                print(f"FROZEN — NOT SUCCESS: {frozen}", flush=True)
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
        time.sleep(max(0.0, 1 / 30 - (time.monotonic() - tick)))
    return 1 if frozen is not None else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW local directory; parent must exist; USD caches and parameter_audit.json")
    parser.add_argument("--max-seconds", type=float, default=3600, help="Wall-clock GUI limit (default: 3600)")
    parser.add_argument("--hand-gravity", choices=("off", "on"), default="off", help="Default off; on adds root-only ideal net weight support, NOT arm dynamics")
    args = parser.parse_args(argv)  # --help imports no numpy/torch/Isaac.
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    probe = load_module("hand2_live_probe", Path(__file__).with_name("validate_hand2_free_space.py"))
    factory = load_module("hand2_live_factory", probe.ROBOTS / "free_hand2_right_hand.py")
    probe.preflight(factory.contract)
    data = Path(os.environ.setdefault("REGRIND_DATA_DIR", str(REPO / "data")))
    if not data.is_dir() or factory.contract.safe_path(data / "hand2_right/bowl/retargeted_120fps.h5") != probe.REFERENCE:
        raise ValueError("REGRIND_DATA_DIR must resolve to the preflight-pinned task data")
    output = factory.contract.new_path(args.output_dir)
    output.mkdir()
    os.environ["HAND2_USD_CACHE_DIR"] = str(output / "hand_usd_cache")
    app = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=False, device="cuda:0").app
        return run(app, probe, factory, args.max_seconds, output, args.hand_gravity)
    finally:
        if app is not None:
            app.close(skip_cleanup=True)  # Only the app created by this invocation.


if __name__ == "__main__":
    raise SystemExit(main())
