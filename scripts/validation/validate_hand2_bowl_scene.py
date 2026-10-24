#!/usr/bin/env python3
"""One bounded actual-task scene check: 497 reference + 120 dynamic frames.

The MP4 is reference replay followed by zero-residual dynamics, not a trained
policy. Exit zero means measurements completed, never physics approval.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import traceback

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parents[1]
TASK_ID = "Regrind-WujiHand2-Bowl-S-v0"
REFERENCE_FRAMES = 497
DYNAMIC_STEPS = 120
MAX_VIDEO_FRAMES = REFERENCE_FRAMES + DYNAMIC_STEPS


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_summary(path, report):
    with path.open("w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_scene(gym_env, data_dir, report, writer):
    # All simulator/runtime imports follow AppLauncher creation in main().
    import h5py
    import numpy as np
    import torch
    import omni.usd
    from PIL import Image, ImageDraw
    from isaaclab.utils.math import quat_error_magnitude
    from regrind.robots.free_hand2_right_hand import contract, inspect_collision_filters
    from regrind.tasks.manager_based.dexterous.config.wujihand2.hand2_bowl_s_env_cfg import (
        CONTROLLER_HYPOTHESIS, PAD_BODIES, TABLE_SENSOR_NAMES, TIP_BODIES,
    )

    env = gym_env.unwrapped
    robot, obj = env.scene["robot"], env.scene["object"]
    command = env.command_manager.get_term("motion")
    ids, joint_names = robot.find_joints(list(contract.JOINT_NAMES), preserve_order=True)
    if (list(joint_names) != list(contract.JOINT_NAMES) or len(robot.joint_names) != 20
            or sorted(ids) != list(range(20)) or robot.is_fixed_base or robot.num_instances != 2):
        raise ValueError("Wrong actual Hand2 floating articulation/joint mapping")
    if env.action_manager.total_action_dim != 26 or gym_env.unwrapped.single_action_space.shape != (26,):
        raise ValueError("Expected actual flattened 26-action space")
    if tuple(env.action_space.shape) != (2, 26):
        raise ValueError("Expected batched action shape (2, 26)")
    if env.action_manager.get_term("root_pose")._wrench_body_ids != [robot.body_names.index("r_mount")]:
        raise ValueError("Controller wrench is not applied to r_mount")
    if set(robot.body_names) != set(contract.model_layout()[0]) or len(robot.body_names) != 33:
        raise ValueError("PhysX must preserve all 33 Hand2 frames")
    if command.target_fingertips_pos.shape != (2, 5, 3) or command.current_fingertips_pos.shape != (2, 5, 3):
        raise ValueError("Actual command/frame-transformer must have five fingertips")
    if command.motion.total_demo_len != REFERENCE_FRAMES or env.max_episode_length != REFERENCE_FRAMES:
        raise ValueError("MotionLoader must retain exactly 497 native-speed control frames")
    if command.cfg.rsi_enabled or command.cfg.traj_aug_enabled or tuple(env.cfg.sim.gravity) != (0.0, 0.0, -9.81):
        raise ValueError("Fixed frame-zero/full-gravity pilot contract changed")
    active_terms = set(env.termination_manager.active_terms)
    required_terms = {"time_out", "demo_end_reached", "object_deviation", "object_out_of_bounds", "hand_far_from_object", "hit_table"}
    if not required_terms.issubset(active_terms):
        raise ValueError("Training terminations are missing")
    for name, body in zip(TABLE_SENSOR_NAMES, PAD_BODIES):
        sensor = env.scene.sensors[name]
        if list(sensor.body_names) != [body] or sensor.data.force_matrix_w is None:
            raise ValueError(f"Filtered pad/table sensor not active: {name}")
        if tuple(sensor.data.force_matrix_w.shape) != (2, 1, 1, 3):
            raise ValueError(f"Unexpected pad/table filter shape: {name}")
    table_top = env.cfg.scene.table.init_state.pos[2] + env.cfg.scene.table.spawn.size[2] / 2
    if abs(table_top - 0.012) > 1e-12:
        raise ValueError("Table geometry changed")
    stage = omni.usd.get_context().get_stage()
    filters = []
    for index in range(2):
        root = f"/World/envs/env_{index}/Robot"
        if stage.GetPrimAtPath(root).IsInstance():
            raise ValueError("Training clone is unexpectedly instanceable")
        info = inspect_collision_filters(stage, root)
        if len(info["actual_prim_pairs"]) != 41:
            raise ValueError("Expected exactly 41 explicit own-robot link filters")
        filters.append(len(info["actual_prim_pairs"]))
    # Check actual PhysX gains and finite effort/velocity/armature, not only config.
    for field, getter in (("kp", "get_dof_stiffnesses"), ("kv", "get_dof_dampings"),
                          ("effort", "get_dof_max_forces"), ("velocity", "get_dof_max_velocities"),
                          ("armature", "get_dof_armatures")):
        actual = getattr(robot.root_physx_view, getter)()[:, ids].detach().cpu().numpy()
        expected = np.array([p[field] for p in contract.PARAMETERS.values()])
        if actual.shape != (2, 20) or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=2e-6, atol=1e-8):
            raise ValueError(f"Applied vendor parameters mismatch: {field}")
    effort = robot.data.joint_effort_limits[:, ids]
    if not torch.isfinite(effort).all() or not (effort > 0).all():
        raise ValueError("Invalid effort limits")
    with h5py.File(data_dir / "retargeted_120fps.h5") as file:
        ref = {k: file[k][()] for k in ("robot_pos", "robot_quat", "robot_joints", "object_pos", "object_quat",
                                       "robot_keypoints", "anchor_wrist_pos", "anchor_wrist_quat")}
        keypoint_names = [n.decode() for n in file["robot_keypoint_link_names"][()]]
    body_ids = {n: robot.body_names.index(n) for n in robot.body_names}
    origins = env.scene.env_origins
    zeros6 = torch.zeros((2, 6), device=env.device)
    frames_written = 0

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=env.device)

    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return bool(torch.isfinite(value).all())

    def refresh():
        env.sim.forward()  # Kinematics only, never a physics step.
        env.scene.update(env.physics_dt)

    def hardset(frame, offset=None):
        root = tensor(np.concatenate((ref["robot_pos"][frame], ref["robot_quat"][frame]))).repeat(2, 1)
        root[:, :3] += origins
        if offset is not None:
            root[:, :3] += offset
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(zeros6)
        q = tensor(ref["robot_joints"][frame]).repeat(2, 1)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
        state = torch.cat((tensor(np.concatenate((ref["object_pos"][frame], ref["object_quat"][frame]))).repeat(2, 1), zeros6), dim=1)
        state[:, :3] += origins
        obj.write_root_state_to_sim(state)
        command.time_steps[:] = frame // 4
        refresh()

    def capture(label):
        nonlocal frames_written
        if frames_written >= MAX_VIDEO_FRAMES:
            raise ValueError("Explicit video frame cap exceeded")
        rgb = env.render()
        if rgb is None or rgb.shape != (480, 640, 3) or not np.isfinite(rgb).all() or not np.any(rgb):
            raise ValueError("Actual training scene returned an invalid/empty RGB frame")
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 38), fill=(0, 0, 0))
        draw.text((8, 3), "HAND2/BOWL reference replay (not policy trained)", fill="white")
        draw.text((8, 20), label, fill="white")
        writer.append_data(np.asarray(image))
        frames_written += 1

    report.update({"action_dim": 26, "joint_names": joint_names, "filter_pairs_per_clone": filters,
                   "controller_hypothesis": CONTROLLER_HYPOTHESIS, "table_top_m": table_top,
                   "table_center_xy": [-0.15, 0.0], "gravity": list(env.cfg.sim.gravity),
                   "terminations": sorted(active_terms), "table_contact_force_threshold_N": 10.0})
    # Asymmetric root positions expose accidental cross-environment resets.
    offset = tensor([[0.0, 0.0, 0.0], [0.03, -0.02, 0.01]])
    hardset(0, offset)
    before = (robot.data.body_link_pose_w[1].clone(), robot.data.joint_pos[1].clone(), obj.data.root_state_w[1].clone())
    env._reset_idx(torch.tensor([0], device=env.device))
    refresh()
    after = (robot.data.body_link_pose_w[1], robot.data.joint_pos[1], obj.data.root_state_w[1])
    if any(not torch.allclose(a, b, atol=1e-6, rtol=0) for a, b in zip(before, after)):
        raise ValueError("Resetting env0 changed asymmetric env1")
    report["reset_isolation"] = {"env1_root_translation_m": [0.03, -0.02, 0.01], "env1_unchanged": True}

    # Warm up only rendering; no simulation step or reference contact claim.
    hardset(0)
    for _ in range(4):
        env.render()
    max_fk = max_rotation = max_tf = 0.0
    checked = 0
    for frame in range(0, 1985, 4):
        hardset(frame)
        if frame in (0, 244, 992, 1588, 1984):
            actual = robot.data.body_link_pose_w.detach().cpu().numpy()
            origins_np = origins.detach().cpu().numpy()
            for index in range(2):
                poses = actual[index].copy()
                poses[:, :3] -= origins_np[index]
                for body, expected in (("r_mount", np.r_[ref["robot_pos"][frame], ref["robot_quat"][frame]]),
                                       ("r_wrist", np.r_[ref["anchor_wrist_pos"][frame], ref["anchor_wrist_quat"][frame]])):
                    pos, rot = contract.check_pose(poses[body_ids[body]].tolist(), expected.tolist())
                    max_fk, max_rotation = max(max_fk, pos), max(max_rotation, rot)
                error = np.linalg.norm(poses[[body_ids[n] for n in keypoint_names], :3] - ref["robot_keypoints"][frame], axis=-1).max()
                max_fk = max(max_fk, float(error))
                if error > contract.POSITION_TOLERANCE_M:
                    raise ValueError("Canonical named FK differs from Drake H5")
            tf = command.current_fingertips_pos
            expected = tensor(ref["robot_keypoints"][frame, [4, 8, 12, 16, 20]])
            error = float(torch.linalg.vector_norm(tf - expected, dim=-1).max())
            max_tf = max(max_tf, error)
            if error > contract.POSITION_TOLERANCE_M or not torch.allclose(command.target_fingertips_pos, expected.expand(2, -1, -1), atol=1e-6, rtol=0):
                raise ValueError("Five fingertip TF/command ordering mismatch")
            if not torch.allclose(robot.data.joint_pos[:, ids], tensor(ref["robot_joints"][frame]).repeat(2, 1), atol=1e-6, rtol=0):
                raise ValueError("Named joints were not set correctly")
            checked += 1
        capture(f"REFERENCE hard-set, no dynamics/contact approval | {frame // 4 + 1}/497")
    report["reference"] = {"frames": REFERENCE_FRAMES, "fps": 30, "duration_s": REFERENCE_FRAMES / 30,
                           "physics_steps": 0, "selected_fk_frames_checked_per_env": checked,
                           "max_fk_position_error_m": max_fk, "max_rotation_error_rad": max_rotation,
                           "max_five_tip_tf_error_m": max_tf, "contact_approval": False}

    # Reset once through the real task; all subsequent writes/resets belong to
    # env.step's normal manager flow, never this validator's hard-set helper.
    observation, _ = gym_env.reset(seed=42)
    if not finite(observation):
        raise ValueError("Nonfinite initial observations")
    values = {key: [] for key in ("root_position_error_m", "root_rotation_error_rad", "anchor_wrist_error_m",
                                  "finger_joint_rmse_rad", "object_tracking_error_m", "object_displacement_from_initial_m",
                                  "effort_saturation_fraction", "max_effort_ratio", "table_force_max_N")}
    termination_counts = {name: [0, 0] for name in env.termination_manager.active_terms}
    reset_counts = torch.zeros(2, dtype=torch.long, device=env.device)
    original_compute = env.termination_manager.compute

    def measured_compute():
        # Capture post-physics/pre-reset state. Delegate termination unchanged;
        # this avoids misreporting an auto-reset pose as successful tracking.
        result = original_compute()
        for name in termination_counts:
            flags = env.termination_manager.get_term(name)
            for index, flag in enumerate(flags.detach().cpu().tolist()):
                termination_counts[name][index] += int(flag)
        states = (robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel,
                  robot.data.applied_torque, obj.data.root_state_w)
        if not finite(states):
            raise ValueError("Nonfinite pre-reset dynamic state/torque")
        frame_indices = (command.time_steps * 4).detach().cpu().numpy()
        anchor = tensor(ref["anchor_wrist_pos"][frame_indices])
        root_error = torch.linalg.vector_norm(command.current_hand_wrist_pos - command.target_hand_wrist_pos, dim=-1)
        rotation_error = quat_error_magnitude(command.current_hand_wrist_quat, command.target_hand_wrist_quat)
        wrist_error = torch.linalg.vector_norm(robot.data.body_link_pose_w[:, body_ids["r_wrist"], :3] - origins - anchor, dim=-1)
        joint_error = ((command.current_hand_joint_pos - command.target_hand_joint_pos) ** 2).mean(dim=-1).sqrt()
        object_error = torch.linalg.vector_norm(command.current_object_pos - command.target_object_pos, dim=-1)
        drift = torch.linalg.vector_norm(command.current_object_pos - tensor(ref["object_pos"][0]), dim=-1)
        ratio = robot.data.applied_torque[:, ids].abs() / effort
        forces = torch.stack([env.scene.sensors[n].data.force_matrix_w[:, 0, 0, :] for n in TABLE_SENSOR_NAMES])
        # The shared hit_table term treats NaN (no contact) as zero; retain that
        # documented convention, but never accept infinities.
        if torch.isinf(forces).any():
            raise ValueError("Infinite table sensor force")
        force_max = torch.linalg.vector_norm(torch.nan_to_num(forces, nan=0.0), dim=-1).max()
        metrics = (root_error, rotation_error, wrist_error, joint_error, object_error, drift,
                   (ratio >= 0.99).float().mean(), ratio.max(), force_max)
        if not finite(metrics):
            raise ValueError("Nonfinite dynamic measurement")
        for key, value in zip(values, metrics):
            values[key].extend(value.reshape(-1).detach().cpu().tolist())
        return result

    env.termination_manager.compute = measured_compute
    try:
        for step in range(DYNAMIC_STEPS):
            action = torch.zeros((2, 26), device=env.device)
            if not finite(action):
                raise ValueError("Nonfinite action")
            observation, reward, terminated, truncated, _ = gym_env.step(action)
            if not finite(observation) or not finite(reward):
                raise ValueError("Nonfinite dynamic observations/rewards")
            reset_counts += (terminated | truncated).long()
            capture(f"DYNAMICS zero residual, full gravity, terminations active | {step + 1}/120")
    finally:
        env.termination_manager.compute = original_compute
    report["dynamics"] = {
        "control_steps": DYNAMIC_STEPS, "physics_steps": DYNAMIC_STEPS * env.cfg.decimation,
        "duration_s": DYNAMIC_STEPS * env.step_dt, "zero_residual_actions": True,
        "external_state_writes_during_steps": 0, "all_terminations_active": True,
        "finite_observations_actions_rewards_states": True,
        "measurement_timing": "post_physics_pre_reset_against_current_command",
        "wrist_target_semantics": "motion wrist is r_mount; anchor_wrist_error measures physical r_wrist",
        "reset_counts_per_env": reset_counts.cpu().tolist(), "termination_counts_per_env": termination_counts,
        "metrics": {k: {"mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in values.items()},
    }
    report["video_frames"] = frames_written
    if frames_written != MAX_VIDEO_FRAMES:
        raise ValueError("Incomplete reference/dynamic video")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New caller-owned artifact directory; never overwritten")
    parser.add_argument("--num-envs", type=int, choices=(2,), default=2)
    args = parser.parse_args(argv)  # --help exits before any Isaac import/launch.
    publisher = load_module("hand2_bowl_publisher", WORKSPACE / "scripts/tactidex_import/retarget/publish_hand2_bowl.py")
    contract = publisher.load_contract()
    output = contract.new_path(args.output_dir)
    output.mkdir()
    summary = output / "summary.json"
    video = output / "scene.mp4"
    report = {"status": "failed", "approved_physics": False, "controller_validation": "pending",
              "runtime_executed": False, "task_id": TASK_ID, "device": "cuda:0", "num_envs": args.num_envs,
              "headless": True, "enable_cameras": True, "video_path": str(video),
              "video_fps": 30, "max_video_frames": MAX_VIDEO_FRAMES,
              "video_description": "497 reference hard-set frames + 120 zero-residual dynamic frames; not trained policy",
              "shutdown": "app.close(skip_cleanup=True); durable artifacts precede shutdown"}
    app = writer = None
    status = 1
    try:
        data_dir = REPO / "data/hand2_right/bowl"
        manifest = publisher.validate_publication(data_dir)
        report["source_hashes"] = {s["relative_path"]: s["sha256"] for s in manifest["sources"]}
        report["runtime_data_hashes"] = manifest["output_sha256"]
        report["manifest_sha256"] = publisher.sha256(data_dir / "manifest.json")
        os.environ["HAND2_USD_CACHE_DIR"] = str(output / "usd_cache")
        os.environ["HAND2_NUM_ENVS"] = str(args.num_envs)
        os.environ["REGRIND_DATA_DIR"] = str(REPO / "data")
        save_summary(summary, report)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
        report["runtime_executed"] = True
        import gymnasium as gym
        import imageio.v2 as imageio
        import regrind.tasks  # registered task discovery, after Simulator startup
        from isaaclab_tasks.utils import parse_env_cfg
        cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=args.num_envs)
        cfg.viewer.resolution = (640, 480)
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        gym_env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array")
        gym_env.reset(seed=42)
        writer = imageio.get_writer(str(video), fps=30, codec="libx264", macro_block_size=16)
        run_scene(gym_env, data_dir, report, writer)
        writer.close()
        writer = None
        with video.open("rb") as stream:
            os.fsync(stream.fileno())
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"], check=True, timeout=30, capture_output=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height", "-of", "json", str(video)],
                               check=True, timeout=30, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        if int(stream["nb_read_frames"]) != MAX_VIDEO_FRAMES or stream["avg_frame_rate"] != "30/1":
            raise ValueError("Decoded video frame count/FPS mismatch")
        report["video_decoded"] = stream
        report["video_sha256"] = publisher.sha256(video)
        report["status"] = "measurements_complete_pending_physics_review"
        status = 0
    except (Exception, SystemExit) as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as error:
                report["video_close_error"] = str(error)
        report["application_close"] = "pending" if app is not None else "not_created"
        save_summary(summary, report)
        fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        # Do not env.close()/sim.stop(): this standalone validator owns the app,
        # and the established immediate path avoids the known cleanup hang.
        if app is not None:
            try:
                app.close(skip_cleanup=True)
            except Exception as error:
                report["application_close"] = "failed"
                report["status"] = "failed"
                report["close_error"] = f"{type(error).__name__}: {error}"
                status = 1
            else:
                report["application_close"] = "closed"
            # SystemExit/process termination intentionally leaves durable pending.
            save_summary(summary, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
