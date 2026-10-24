#!/usr/bin/env python3
"""Interactive actual Hand2 Bowl task, zero residual actions, NO POLICY.

Uncalibrated inertia-derived angular PD; vendor fingers, gravity, table and task
terminations are unchanged. Bowl moves by dynamics except normal episode resets.
Run in the existing Isaac environment with a NEW caller-owned --output-dir.
"""
import argparse
import importlib.util
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


def install_root_guard(term, probe, array, native=native_pd):
    original = term.apply_actions

    def apply_guarded(self):
        try:
            expected = probe.root_controller(PROFILE)
            if any(getattr(self, key) != expected[key] for key in ("kp_pos", "kd_pos", "kp_rot", "kd_rot")):
                raise ValueError("Runtime root gains differ from guarded profile")
            guarded_pd(self, probe, array)
            # Check exact native tensor arithmetic too; no clipping/replacement.
            probe.pd_wrench_guard(tuple(array(value)[0] for value in native(self)))
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


def audit_runtime(env, factory):
    """Read-only live masses and actual rigid-body USD gravity/kinematic flags."""
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdPhysics, PhysxSchema
    mass = factory.check_runtime_masses(env.scene["robot"])
    stage = omni.usd.get_context().get_stage()
    for name, disable, count in (("Robot", True, 33), ("Object", False, 1)):
        root = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        if not root.IsValid():
            raise ValueError(f"Missing {name} prim")
        bodies = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != count:
            raise ValueError(f"Unexpected {name} rigid-body count: {len(bodies)}")
        for prim in bodies:
            gravity = PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()
            body = UsdPhysics.RigidBodyAPI(prim)
            if gravity is not disable or body.GetKinematicEnabledAttr().Get() or not body.GetRigidBodyEnabledAttr().Get():
                raise ValueError(f"Unexpected gravity/kinematic/body-enabled flag: {prim.GetPath()}")
    object_mass = env.scene["object"].root_physx_view.get_masses().detach().cpu().numpy()
    if object_mass.size != 1 or not np.isfinite(object_mass).all() or not (object_mass > 0).all():
        raise ValueError("Invalid live Bowl mass")
    return mass["total_mass_kg_per_env"][0], float(object_mass.reshape(-1)[0])


def ui_update_only(app, sim):
    # IsaacLab render() uses this setting too: update UI without a physics tick,
    # even if the user presses Play during a latched failure.
    sim.set_setting("/app/player/playSimulations", False)
    try:
        app.update()
    finally:
        sim.set_setting("/app/player/playSimulations", True)


def run(app, probe, factory, max_seconds):
    import gymnasium as gym
    import numpy as np
    import torch
    import omni.ui as ui
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import regrind.tasks  # noqa: F401

    window = ui.Window("DYNAMIC BOWL — NO POLICY", width=650, height=180)
    with window.frame:
        with ui.VStack(spacing=3):
            ui.Label("Bowl: gravity + contact only; normal episode resets; NO per-frame Bowl writes")
            ui.Label("Hand gravity OFF | Bowl gravity ON | table top 0.012 m; known initial overlap")
            ui.Label("Controller UNCALIBRATED | inertia-derived root angular PD | vendor fingers unchanged")
            status = ui.Label("STARTING: runtime mass/gravity checks pending", word_wrap=True)
            motion_status = ui.Label("Object actual/reference position error: pending", word_wrap=True)
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1, use_fabric=True)
    configure_root(cfg, probe)
    if cfg.scene.num_envs != 1 or cfg.decimation != 4 or not math.isclose(cfg.sim.dt, 1 / 120):
        raise ValueError("Expected one environment, 120 Hz physics / 30 Hz actions")
    if not math.isclose(cfg.scene.table.init_state.pos[2] + cfg.scene.table.spawn.size[2] / 2, 0.012):
        raise ValueError("Unexpected table height")
    wrapped = gym.make(TASK, cfg=cfg)
    env = wrapped.unwrapped
    sim = env.sim
    array = lambda value: value.detach().cpu().numpy()
    # gym.make initializes PhysX; audit BEFORE gym.reset or any env.step.
    hand_mass, bowl_mass = audit_runtime(env, factory)
    term = env.action_manager.get_term("root_pose")
    install_root_guard(term, probe, array)
    if env.num_envs != 1 or env.action_manager.total_action_dim != 26:
        raise ValueError("Expected single-environment 26-D action manager")
    obs, _ = wrapped.reset()  # The only explicit reset; task owns episode auto-resets.
    actions = torch.zeros((1, 26), device=env.device)
    command = env.command_manager.get_term("motion")
    steps, resets, frozen = 0, 0, None
    start = last_print = time.monotonic()
    print(f"READY task={TASK} NO_POLICY kp_pos=300 kd_pos=30 kp_rot={term.kp_rot} kd_rot={term.kd_rot} "
          f"hand_mass_kg={hand_mass} bowl_mass_kg={bowl_mass} hand_gravity=OFF bowl_gravity=ON "
          "table_top_m=0.012 UNCALIBRATED known_table_overlap=true no_per_frame_Bowl_writes=true", flush=True)
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
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW local directory; parent must exist; USD caches only")
    parser.add_argument("--max-seconds", type=float, default=3600, help="Wall-clock GUI limit (default: 3600)")
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
        return run(app, probe, factory, args.max_seconds)
    finally:
        if app is not None:
            app.close(skip_cleanup=True)  # Only the app created by this invocation.


if __name__ == "__main__":
    raise SystemExit(main())
