#!/usr/bin/env python3
"""Scissors-only actual scene audit, NOT POLICY or a physics certificate.

Default/--check is sim-free/read-only. --run requires the exact existing central
record, fresh scoped video, published approved bundle and caller-owned cache.
No training, source edits, hidden simulation, extra snapshots or task relaxation.
"""
import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
REPO = ROOT / "repos/regrind"
CONTRACT_PATH = REPO / "source/regrind/regrind/robots/hand2_scissors_contract.py"
_spec = importlib.util.spec_from_file_location("scissors_scene_contract", CONTRACT_PATH)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
require = C.require
RECORD = ROOT / "REGRIND/artifacts/hand2_right/mesh_tip_ppo_check_v1/run.json"
VIDEO = REPO / "logs/validation/wujihand2_scissors/20261028_scene/zero_residual_scene.mp4"
KEY = "scissors_actual_scene_20261028"
TASK = "Regrind-WujiHand2-Scissors-Official-S-v0"
TERMS = ("time_out", "demo_end_reached", "object_deviation", "object_out_of_bounds", "hand_far_from_object", "hit_table")
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
SENSORS = tuple(f"contact_hand2_{f}_table" for f in FINGERS)
CONTROLS, TIMEOUT = C.CONTROLS, 1800


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def finite(value, name="value"):
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, name+"/"+key)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item, name)
    else:
        require(np.isfinite(array(value)).all(), f"Nonfinite {name}")


def precheck():
    spec = importlib.util.spec_from_file_location("scissors_scene_publisher", ROOT / "scripts/arctic_hand2/publish_scissors.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    _, snapshot = publisher.check_sources()
    # Include this implementation and the actual runtime code in before/after evidence.
    paths = [CONTRACT_PATH, Path(__file__), ROOT / "scripts/arctic_hand2/publish_scissors.py",
             REPO / "source/regrind/regrind/robots/free_hand2_right_hand.py",
             REPO / "source/regrind/regrind/objects/scissors.py"]
    base = REPO / "source/regrind/regrind/tasks/manager_based/dexterous"
    paths += [base / "dexterous_env_cfg.py", *(base / "mdp" / n for n in ("commands.py", "actions.py", "hand2_supported_root.py", "terminations.py", "rewards.py", "observations.py"))]
    paths += list((base / "config/wujihand2_scissors").glob("*.py"))
    snapshot.update({p.relative_to(ROOT).as_posix(): C.sha256(p) for p in paths})
    bundle = C.bundle_path(REPO / "data")
    if bundle.exists():
        C.validate_bundle(bundle)
        snapshot.update({p.relative_to(ROOT).as_posix(): C.sha256(p) for p in bundle.rglob("*") if p.is_file()})
    return {"source_hashes_before": snapshot, "bundle_present": bundle.exists(), "native_frames": C.FRAMES,
            "runtime_controls": CONTROLS, "physics_certified": False, "source_check_passed": True}


def verify_snapshot(snapshot):
    for relative, digest in snapshot.items():
        C.verify_hash(C.local_path(ROOT, relative), digest)
    return dict(snapshot)


def validate_outputs(video, record, cache):
    video, record, cache = (C.safe_path(p) for p in (video, record, cache))
    require(video == VIDEO and record == RECORD, "Only the exact Scissors video and existing central record are allowed")
    require(not video.exists() and video.parent.is_dir(), "Fresh video and existing parent required")
    C.regular_file(record)
    require(KEY not in json.loads(record.read_text()), "Scene record already exists")
    require(cache.parent == Path("/tmp") and cache.name.startswith("hand2-scissors-") and cache.is_dir()
            and cache.stat().st_uid == os.getuid() and not (cache / "hand_usd").exists(), "Caller-owned existing /tmp/hand2-scissors-* cache with fresh hand_usd required")
    return video, record, cache


def flush_record(path, report):
    """Append only our key under flock, rereading latest unrelated keys, then fsync."""
    path = C.regular_file(path)
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(fd, "r+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        st = path.stat()
        require((st.st_dev, st.st_ino) == (os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino), "Central record replaced")
        current = json.load(stream)
        require(KEY not in current, "Refusing existing scene record key")
        current[KEY] = report
        payload = json.dumps(current, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
        stream.seek(0)
        stream.write(payload)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())


class StateWriteAudit:
    """Count nested public state setters; only startup and ordinary resample write state."""
    def __init__(self):
        self.phase, self.depth, self.control = "startup", 0, 0
        self.records, self.reset_calls, self.resample_calls = [], [], []

    @contextmanager
    def section(self, phase):
        before = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = before

    def install_asset(self, asset, label):
        methods = [n for n in dir(asset) if n.startswith("write_") and n.endswith("_to_sim")
                   and any(n == f"write_joint_{kind}_to_sim" for kind in ("state", "position", "velocity"))]
        methods += [n for n in dir(asset) if n.startswith("write_root_") and n.endswith("_to_sim")]
        require(methods, f"No actual state setters: {label}")
        targets = [(asset, name, "Articulation."+name) for name in sorted(set(methods))]
        for name in ("set_root_transforms", "set_root_velocities", "set_dof_positions", "set_dof_velocities"):
            require(callable(getattr(asset.root_physx_view, name, None)), f"Missing actual PhysX state setter: {name}")
            targets.append((asset.root_physx_view, name, "PhysX."+name))
        for target, name, api in targets:
            original = getattr(target, name)
            def write(*args, _original=original, _name=api, **kwargs):
                record = {"asset": label, "setter": _name, "phase": self.phase,
                          "nested_depth": self.depth, "control": self.control}
                self.records.append(record)
                require(self.phase in ("startup", "ordinary_resample_command"), "Added state teleport outside startup/ordinary MotionCommand resample")
                self.depth += 1
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.depth -= 1
            setattr(target, name, write)

    def install_reset(self, env, command):
        reset, resample = env._reset_idx, command._resample_command
        def ordinary_reset(ids):
            self.reset_calls.append({"control": self.control, "env_ids": array(ids).tolist(), "from_phase": self.phase})
            with self.section("ordinary_reset"):
                return reset(ids)
        def ordinary_resample(ids):
            require(self.phase == "ordinary_reset", "Added command resample outside ordinary environment reset")
            self.resample_calls.append({"control": self.control, "env_ids": array(ids).tolist()})
            with self.section("ordinary_resample_command"):
                return resample(ids)
        env._reset_idx, command._resample_command = ordinary_reset, ordinary_resample

    def summary(self):
        counts = {}
        for r in self.records:
            key = r["phase"]+"/"+r["asset"]+"/"+r["setter"]
            counts[key] = counts.get(key, 0)+1
        return {"records": self.records, "counts": counts, "reset_calls": self.reset_calls,
                "resample_calls": self.resample_calls, "nested_calls_counted": True,
                "added_rollout_state_writes": sum(r["phase"] not in ("startup", "ordinary_resample_command") for r in self.records)}


def force_identity(forces, torques, positions, masses, coms, root, pd_force, pd_torque):
    """Independent float64 identity using APPLIED capped PD, not requested force."""
    force, torque, point, mass, com, root, pf, pt = [array(v).astype(np.float64) for v in
        (forces, torques, positions, masses, coms, root, pd_force, pd_torque)]
    for x in (force, torque, point, mass, com, root, pf, pt):
        finite(x, "wrench identity")
    require(force.shape == torque.shape == point.shape == (1, 1, 3) and root.shape == pf.shape == pt.shape == (1, 3)
            and mass.ndim == 2 and mass.shape[0] == 1 and com.shape == (*mass.shape, 3) and np.all(mass > 0), "Wrench shape/mass")
    weight = mass[..., None]*np.array([0., 0., -9.81])
    support_force = -weight.sum(axis=1)
    support_torque = -np.cross(com-root[:, None], weight).sum(axis=1)
    whole_com = (mass[..., None]*com).sum(axis=1)/mass.sum(axis=1, keepdims=True)
    com_moment = np.cross(whole_com-root, pf)
    force_error = float(np.max(np.abs(force[:, 0]-pf-support_force)))
    torque_error = float(np.max(np.abs(torque[:, 0]-pt-support_torque-com_moment)))
    require(np.max(np.abs(point[:, 0]-root)) <= 1e-7, "Wrench must act at root LINK")
    require(force_error <= 5e-5 and torque_error <= 5e-6, "Applied support/COM wrench identity failed")
    require(np.linalg.norm(pf, axis=1).max() <= 50.+1e-4 and np.linalg.norm(pt, axis=1).max() <= 1.+1e-5,
            "PD component cap failed (not a total wrench cap)")
    return {"force_identity_abs_error_N": force_error, "torque_identity_abs_error_Nm": torque_error,
            "applied_pd_force_N": pf.tolist(), "applied_pd_torque_Nm": pt.tolist(),
            "support_force_N": support_force.tolist(), "support_torque_Nm": support_torque.tolist(),
            "applied_PD_COM_moment_Nm": com_moment.tolist(), "combined_force_N": force[:, 0].tolist(),
            "combined_torque_Nm": torque[:, 0].tolist()}


def zero_object_buffers(obj):
    result = {}
    for name in ("_external_force_b", "_external_torque_b"):
        require(hasattr(obj, name), "Actual object external wrench buffers unavailable")
        value = array(getattr(obj, name))
        require(np.isfinite(value).all() and np.count_nonzero(value) == 0, f"Object compensation detected: {name}")
        result[name] = float(np.max(np.abs(value))) if value.size else 0.
    return result


class WrenchAudit:
    def __init__(self, env, state_audit, pd_function):
        self.env, self.phase, self.pd_function = env, state_audit, pd_function
        self.root_rows, self.root_submissions, self.object_writes, self.physics_steps, self.reset_zero_writes = [], [], [], [], []

    def install(self):
        env, phase = self.env, self.phase
        robot, obj = env.scene["robot"], env.scene["object"]
        root = env.action_manager.get_term("root_pose")
        original = robot.set_external_force_and_torque
        signature = inspect.signature(original)
        def hand_write(*args, **kwargs):
            values = signature.bind(*args, **kwargs)
            values.apply_defaults()
            v = values.arguments
            if phase.phase in ("ordinary_reset", "ordinary_resample_command"):
                require(not np.count_nonzero(array(v["forces"])) and not np.count_nonzero(array(v["torques"])), "Nonzero reset wrench")
                self.reset_zero_writes.append({"control": phase.control, "phase": phase.phase})
                return original(*args, **kwargs)
            require(phase.phase == "rollout", "Unexpected root wrench outside physics rollout")
            ids = array(v["body_ids"]).reshape(-1).tolist()
            require(ids == [robot.body_names.index("r_mount")] and v["is_global"] is True, "One world-space r_mount wrench required")
            step = int(env._sim_step_counter)
            require(not self.root_rows or self.root_rows[-1]["physics_step"] != step, "Duplicate root wrench in one physics step")
            # apply_actions assigns _wrist_v3_last_pd AFTER the setter. Recompute
            # the CURRENT pure PD from the same pre-write state/targets instead.
            diagnostics = {}
            pf, pt = self.pd_function(robot.data.root_state_w, env.scene.env_origins, root.target_pos, root.target_quat,
                (root.kp_pos, root.kd_pos, root.kp_rot, root.kd_rot), profile="com_wrench_bounded_v4", diagnostics=diagnostics)
            row = force_identity(v["forces"], v["torques"], v["positions"], root._body_mass,
                                 robot.data.body_com_pos_w, robot.data.root_link_pos_w, pf, pt)
            row.update(physics_step=step, control=phase.control,
                       force_saturated=bool(array(diagnostics["force_saturated"])[0]),
                       torque_saturated=bool(array(diagnostics["saturated"])[0]),
                       requested_pd_force_N=array(diagnostics["force_requested"]).tolist(),
                       requested_pd_torque_Nm=array(diagnostics["requested"]).tolist())
            self.root_rows.append(row)
            return original(*args, **kwargs)
        robot.set_external_force_and_torque = hand_write
        raw_hand_write = robot.root_physx_view.apply_forces_and_torques_at_position
        def hand_physx_write(force_data, torque_data, position_data, indices, is_global):
            force, torque = array(force_data).reshape(1, -1, 3), array(torque_data).reshape(1, -1, 3)
            finite({"force": force, "torque": torque}, "actual PhysX hand submission")
            if phase.phase in ("ordinary_reset", "ordinary_resample_command", "ordinary_reset_request"):
                require(not np.count_nonzero(force) and not np.count_nonzero(torque), "Nonzero ordinary-reset PhysX hand wrench")
            else:
                step, root_id = int(env._sim_step_counter), robot.body_names.index("r_mount")
                require(phase.phase == "rollout" and is_global is True and self.root_rows
                        and self.root_rows[-1]["physics_step"] == step, "PhysX hand submission without current combined root setter")
                row = self.root_rows[-1]
                require(force.shape == torque.shape == (1, len(robot.body_names), 3), "PhysX hand body inventory")
                require(np.count_nonzero(np.delete(force, root_id, axis=1)) == 0
                        and np.count_nonzero(np.delete(torque, root_id, axis=1)) == 0, "Non-root hand wrench forbidden")
                require(np.allclose(force[:, root_id], row["combined_force_N"], rtol=0, atol=5e-5)
                        and np.allclose(torque[:, root_id], row["combined_torque_Nm"], rtol=0, atol=5e-6)
                        and np.allclose(array(position_data).reshape(1, -1, 3)[:, root_id], array(robot.data.root_link_pos_w), rtol=0, atol=1e-7),
                        "Actual PhysX wrench differs from audited setter")
                self.root_submissions.append({"physics_step": step, "control": phase.control, "root_body_id": root_id})
            return raw_hand_write(force_data, torque_data, position_data, indices, is_global)
        robot.root_physx_view.apply_forces_and_torques_at_position = hand_physx_write
        original_object = obj.set_external_force_and_torque
        object_signature = inspect.signature(original_object)
        def object_write(*args, **kwargs):
            bound = object_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            zero = not np.count_nonzero(array(values["forces"])) and not np.count_nonzero(array(values["torques"]))
            self.object_writes.append({"phase": phase.phase, "control": phase.control, "zero": zero, "api": "set_external_force_and_torque"})
            require(zero and phase.phase in ("ordinary_reset", "ordinary_resample_command"), "Added object external wrench write")
            return original_object(*args, **kwargs)
        obj.set_external_force_and_torque = object_write
        # Observe the actual PhysX submission as well as the public setter/buffers.
        view = obj.root_physx_view
        raw_write = view.apply_forces_and_torques_at_position
        def object_physx_write(*args, **kwargs):
            force = kwargs.get("force_data", args[0] if args else None)
            torque = kwargs.get("torque_data", args[1] if len(args) > 1 else None)
            require(force is not None and torque is not None, "Unknown PhysX object wrench signature")
            zero = np.isfinite(array(force)).all() and np.isfinite(array(torque)).all() and not np.count_nonzero(array(force)) and not np.count_nonzero(array(torque))
            self.object_writes.append({"phase": phase.phase, "control": phase.control, "zero": bool(zero), "api": "PhysX.apply_forces_and_torques_at_position"})
            require(zero, "Direct PhysX object compensation")
            return raw_write(*args, **kwargs)
        view.apply_forces_and_torques_at_position = object_physx_write
        original_step = env.sim.step
        def physics_step(*args, **kwargs):
            require(phase.phase == "rollout", "Added integration outside rollout")
            step = int(env._sim_step_counter)
            require(sum(r["physics_step"] == step for r in self.root_rows) == 1
                    and sum(r["physics_step"] == step for r in self.root_submissions) == 1,
                    "Exactly one combined root setter AND actual PhysX submission per physics step required")
            buffers = zero_object_buffers(obj)
            self.physics_steps.append({"physics_step": step, "control": phase.control, "object_buffers": buffers})
            return original_step(*args, **kwargs)
        env.sim.step = physics_step

    def summary(self):
        return {"root_writes": self.root_rows, "actual_root_PhysX_submissions": self.root_submissions,
                "physics_steps": self.physics_steps, "object_writes": self.object_writes,
                "ordinary_reset_zero_root_writes": self.reset_zero_writes, "root_write_count": len(self.root_rows),
                "physics_step_count": len(self.physics_steps), "object_nonzero_writes": sum(not r["zero"] for r in self.object_writes)}


def install_termination_capture(env, rows, sample):
    """Capture each actual callable once per compute, before ordinary resets."""
    manager = env.termination_manager
    require(env.num_envs == 1 and set(manager.active_terms) == set(TERMS), "Exactly six normal termination terms required")
    terms = [(name, manager.get_term_cfg(name)) for name in TERMS]
    original = manager.compute
    current = {}
    for name, cfg in terms:
        func = cfg.func
        require(bool(cfg.time_out) == (name == "time_out"), "Normal timeout classification changed")
        def term(*args, _name=name, _func=func, **kwargs):
            result = _func(*args, **kwargs)
            require(_name not in current and array(result).shape == (1,), "Term called more than once/wrong shape")
            current[_name] = bool(array(result)[0])
            return result
        cfg.func = term
    def compute():
        current.clear()
        result = original()
        require(set(current) == set(TERMS), "Missing actual termination callable result")
        terminated = any(current[n] for n in TERMS if n != "time_out")
        truncated = current["time_out"]
        require(terminated == bool(array(manager.terminated)[0]) and truncated == bool(array(manager.time_outs)[0])
                and (terminated or truncated) == bool(array(result)[0]), "Actual terms/manager flags disagree")
        row = {"control": len(rows)+1, "causes": dict(current), "manager_terminated": terminated,
               "manager_truncated": truncated, "computed_done": terminated or truncated}
        rows.append(row)  # Keep partial failure evidence even if the state sample fails.
        row.update(sample())
        return result
    manager.compute = compute


def record_step_result(row, terminated, truncated):
    require(array(terminated).shape == array(truncated).shape == (1,), "Returned step flags shape")
    row["returned_terminated"], row["returned_truncated"] = bool(array(terminated)[0]), bool(array(truncated)[0])
    require(row["returned_terminated"] == row["manager_terminated"] and row["returned_truncated"] == row["manager_truncated"], "Returned step flags differ from pre-reset callable results")


def sample_control(env):
    command = env.command_manager.get_term("motion")
    root = env.action_manager.get_term("root_pose")
    robot, obj = env.scene["robot"], env.scene["object"]
    step = int(command.timesteps_in_current_demo[0])
    require(0 <= step < CONTROLS, "Reference phase out of range")
    pads = {}
    for name in SENSORS:
        matrix = array(env.scene.sensors[name].data.force_matrix_w)
        require(matrix.shape == (1, 1, 1, 3) and not np.isinf(matrix).any(), "Missing/malformed filtered pad forces")
        vector = np.nan_to_num(matrix[0, 0, 0], nan=0.)
        pads[name] = {"normal_force_w_N": vector.tolist(), "norm_N": float(np.linalg.norm(vector)),
                      "nan_as_zero_components": int(np.isnan(matrix).sum())}
    values = {"robot_root_state": robot.data.root_state_w, "finger_pos": robot.data.joint_pos,
              "finger_vel": robot.data.joint_vel, "finger_effort_estimate_Nm": robot.data.applied_torque,
              "object_root_state": obj.data.root_state_w, "object_joint_rad": obj.data.joint_pos,
              "object_joint_velocity": obj.data.joint_vel, "actions": env.action_manager.action,
              "finger_target_error_rad": robot.data.joint_pos_target-robot.data.joint_pos,
              "root_target_error_m": root.target_pos-(robot.data.root_pos_w-env.scene.env_origins),
              "object_position_error_m": command.current_object_pos-command.target_object_pos,
              "object_joint_error_rad": command.current_object_joint-command.target_object_joint}
    finite(values, "pre-reset state/action/effort")
    require(array(values["actions"]).shape == (1, 26) and np.count_nonzero(array(values["actions"])) == 0,
            "Actual audit actions must be exactly zero residual")
    def angle(a, b):
        x, y = array(a)[0], array(b)[0]
        return float((Rotation.from_quat(x, scalar_first=True)*Rotation.from_quat(y, scalar_first=True).inv()).magnitude())
    return {"reference_timestep": step, "reference_phase_fraction": step/(CONTROLS-1),
            "source_native_frame_120fps": step*640/(CONTROLS-1), "source_frame_30fps": 25+step*160/(CONTROLS-1),
            "episode_control": int(env.episode_length_buf[0]), "pads": pads,
            "root_rotation_error_rad": angle(root.target_quat, robot.data.root_quat_w),
            "object_rotation_error_rad": angle(command.current_object_quat, command.target_object_quat),
            **{k: array(v).tolist() for k, v in values.items()}}


def actual_link_poses(asset):
    """PhysX raw xyzw -> wxyz. Never use just-written root/data pose buffers as FK."""
    raw = array(asset.root_physx_view.get_link_transforms()).copy()
    require(raw.shape == (1, len(asset.body_names), 7), "Actual PhysX link transform shape")
    raw[..., 3:7] = raw[..., [6, 3, 4, 5]]
    finite(raw, "actual link transforms")
    return {name: raw[0, i] for i, name in enumerate(asset.body_names)}


def startup_fk(env, bundle, model, report):
    import torch
    robot, obj = env.scene["robot"], env.scene["object"]
    require(set(obj.body_names) == {"bottom", "top"} and len(obj.body_names) == 2 and list(obj.joint_names) == ["scissors_joint"], "Actual Scissors articulation topology")
    ids = [robot.joint_names.index(n) for n in model.JOINT_NAMES]
    require(len(set(ids)) == 20 and len(robot.joint_names) == 20, "Actual official finger order")
    with h5py.File(bundle / "retargeted_120fps.h5", "r") as f:
        arrays = {k: f[k][()] for k in C.SHAPES}
    maximum, minimum = int(np.argmax(arrays["object_joint"])), int(np.argmin(arrays["object_joint"]))
    cases = [(i, False) for i in dict.fromkeys((0, maximum, minimum, 228, 640))] + [(0, True)]
    poses = report.setdefault("startup_fk", [])
    origin = array(env.scene.env_origins)[0]
    for row, synthetic in cases:
        def tensor(a):
            return torch.as_tensor(np.asarray(a)[None], dtype=robot.data.joint_pos.dtype, device=env.device)
        hand_pose = np.r_[arrays["robot_pos"][row]+origin, arrays["robot_quat"][row]]
        bottom_pose = np.r_[arrays["object_pos"][row]+origin, arrays["object_quat"][row]]
        opening = 0. if synthetic else float(arrays["object_joint"][row, 0])
        robot.write_root_pose_to_sim(tensor(hand_pose))
        robot.write_root_velocity_to_sim(tensor(np.zeros(6)))
        robot.write_joint_state_to_sim(tensor(arrays["robot_joints"][row]), tensor(np.zeros(20)), joint_ids=ids)
        obj.write_root_pose_to_sim(tensor(bottom_pose))
        obj.write_root_velocity_to_sim(tensor(np.zeros(6)))
        obj.write_joint_state_to_sim(tensor([opening]), tensor([0.]))
        env.sim.forward()
        env.scene.update(env.physics_dt)  # Refresh timestamps; NO physics integration.
        env.sim.physics_sim_view.update_articulations_kinematic()
        hand, scissors = actual_link_poses(robot), actual_link_poses(obj)
        top_rotation = Rotation.from_quat(bottom_pose[3:], scalar_first=True)*Rotation.from_rotvec([0., 0., -opening])
        expected_top = np.r_[bottom_pose[:3], top_rotation.as_quat(scalar_first=True)]
        expected_wrist = np.r_[arrays["anchor_wrist_pos"][row]+origin, arrays["anchor_wrist_quat"][row]]
        comparisons = {"r_mount": (hand["r_mount"], hand_pose), "r_wrist": (hand["r_wrist"], expected_wrist),
                       "bottom": (scissors["bottom"], bottom_pose), "top": (scissors["top"], expected_top)}
        errors = {name: model.check_pose(a.tolist(), b.tolist()) for name, (a, b) in comparisons.items()}
        actual_q = array(obj.root_physx_view.get_dof_positions())
        hand_q = array(robot.root_physx_view.get_dof_positions())[0, ids]
        require(np.allclose(hand_q, arrays["robot_joints"][row], atol=1e-6, rtol=0)
                and np.allclose(actual_q, [[opening]], atol=1e-6, rtol=0), "Live joints differ after startup hard-set")
        poses.append({"reference_row": row, "synthetic_zero_opening_nonreference": synthetic, "integration_steps": 0,
                      "object_joint_rad": float(actual_q[0, 0]), "actual_link_poses_wxyz": {n: a.tolist() for n, (a, _) in comparisons.items()},
                      "errors_m_rad": errors, "tolerances_m_rad": [model.POSITION_TOLERANCE_M, model.ROTATION_TOLERANCE_RAD]})


def inspect_colliders(stage):
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath("/World/envs/env_0/Object")
    owners = {"/World/envs/env_0/Object/bottom/"+n for n in ("bottom", "top")}
    require(root.IsValid(), "Missing actual object root")
    result, seen, geometry = [], set(), set()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        owner = prim
        while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        owner_path = str(owner.GetPath()) if owner.IsValid() else ""
        require(path not in seen and owner_path in owners, "Duplicate/missing/wrong collider owner")
        require(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True, "Disabled actual collider")
        seen.add(path)
        local = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).tolist()
        world = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).tolist()
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else "none"
        shape = {"type": prim.GetTypeName()}
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points, indices = np.asarray(mesh.GetPointsAttr().Get()), np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            require(points.ndim == 2 and points.shape[1] == 3 and len(points) > 0 and indices.size > 0, "Missing collider mesh geometry")
            finite(points, "collider geometry")
            import hashlib
            shape.update(vertices=len(points), face_indices=len(indices), geometry_sha256=hashlib.sha256(points.tobytes()+indices.tobytes()).hexdigest())
        else:
            shape["attributes"] = {a.GetName(): str(a.Get()) for a in prim.GetAttributes() if not a.GetName().startswith("xform")}
        identity = json.dumps([owner_path, shape, world], sort_keys=True)
        require(identity not in geometry, "Duplicate collider geometry/transform on owner")
        geometry.add(identity)
        result.append({"path": path, "owner": owner_path, "geometry": shape, "approximation": str(approximation),
                       "local_transform": local, "world_transform": world})
    require(result and {r["owner"] for r in result} == owners, "Missing bottom/top CollisionAPI geometry")
    return result  # Deliberately no assertion equating USD colliders with 30 Drake hulls.


def inspect_surface_materials(env, stage, report):
    """Read live shape coefficients AND resolved USD materials; never author physics.

    Scissors joint friction .2 is unrelated to the inherited [1, 1, 0] surface.
    A single-instance PhysX view has no inter-instance shape padding. Its cooked
    shape count is not inferred from authored CollisionAPI prims or Drake hulls.
    """
    from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema
    expected = {"static_friction": 1., "dynamic_friction": 1., "restitution": 0.}
    audit = {"baseline": {**expected, "friction_combine_mode": "average", "restitution_combine_mode": "average"},
             "baseline_source": ["regrind/tasks/manager_based/dexterous/dexterous_env_cfg.py:DexterousEnvCfg.__post_init__",
                                 "isaaclab/sim/spawners/materials/physics_materials_cfg.py:RigidBodyMaterialCfg"],
             "joint_friction_is_not_surface_friction": True, "physics_scene": {}, "materials": {}, "assets": {}}
    report["actual_surface_materials"] = audit  # Retain partial evidence on rejection.
    scene_path = env.sim.cfg.physics_prim_path
    scene = stage.GetPrimAtPath(scene_path)
    require(scene.IsValid() and scene.IsA(UsdPhysics.Scene), "Missing actual PhysicsScene for surface fallback")
    default_path = scene_path+"/defaultMaterial"

    def binding(prim):
        material, relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        record = {"queried_prim": str(prim.GetPath()), "queried_purpose": "physics",
                  "resolved_material": str(material.GetPrim().GetPath()) if material else None,
                  "relationship": str(relationship.GetPath()) if relationship else None,
                  "binding_owner": str(relationship.GetPrim().GetPath()) if relationship else None,
                  "binding_name": relationship.GetName() if relationship else None,
                  "targets": [str(p) for p in relationship.GetTargets()] if relationship else []}
        return material, relationship, record

    def read_material(material):
        prim = material.GetPrim()
        path = str(prim.GetPath())
        entry = audit["materials"].setdefault(path, {"prim_path": path, "usd_values": {}, "physx_combine_modes": {}})
        require(prim.IsValid() and prim.IsA(UsdShade.Material) and prim.HasAPI(UsdPhysics.MaterialAPI)
                and prim.HasAPI(PhysxSchema.PhysxMaterialAPI), "Missing actual USD/PhysX surface material schemas")
        usd, physx = UsdPhysics.MaterialAPI(prim), PhysxSchema.PhysxMaterialAPI(prim)
        for key, getter in (("static_friction", "GetStaticFrictionAttr"), ("dynamic_friction", "GetDynamicFrictionAttr"),
                            ("restitution", "GetRestitutionAttr")):
            value = getattr(usd, getter)().Get()
            entry.setdefault("usd_attribute_raw_repr", {})[key] = repr(value)
            require(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
                    and np.isfinite(value), "Missing/malformed actual USD surface coefficient: "+key)
            entry["usd_values"][key] = float(value)
            require(value == expected[key], "Actual USD surface coefficient differs from inherited baseline: "+key)
        for key, getter in (("friction_combine_mode", "GetFrictionCombineModeAttr"),
                            ("restitution_combine_mode", "GetRestitutionCombineModeAttr")):
            value = getattr(physx, getter)().Get()
            entry["physx_combine_modes"][key] = str(value) if value is not None else None
            require(value == "average", "Actual PhysX material combine mode differs from average: "+key)
        require(path == default_path, "Surface material identity differs from inherited PhysicsScene/defaultMaterial")
        return path

    scene_material, scene_rel, scene_binding = binding(scene)
    audit["physics_scene"] = scene_binding
    require(scene_material and scene_rel and scene_binding["binding_owner"] == scene_path
            and scene_binding["binding_name"] == "material:binding:physics"
            and scene_binding["targets"] == [default_path] and scene_binding["resolved_material"] == default_path,
            "Missing/malformed actual PhysicsScene default physics material binding")
    read_material(scene_material)
    for name, root_path in (("table", "/World/envs/env_0/Table"), ("object", "/World/envs/env_0/Object")):
        asset = env.scene[name]
        view = asset.root_physx_view
        entry = audit["assets"][name] = {"root_prim": root_path, "bindings": [], "live_api": "get_material_properties",
                                        "coefficient_order": list(expected), "shape_count_is_not_authored_collider_count": True}
        getter = getattr(view, "get_material_properties", None)
        require(callable(getter), "Missing actual PhysX surface getter: "+name)
        properties = array(getter())
        entry.update(view_count=view.count, physx_max_shapes=view.max_shapes,
                     actual_tensor_shape=list(properties.shape), actual_tensor_dtype=str(properties.dtype))
        require(view.count == 1 and isinstance(view.max_shapes, (int, np.integer)) and view.max_shapes > 0
                and properties.shape == (1, view.max_shapes, 3) and properties.dtype.kind in "fi",
                "Malformed actual PhysX surface tensor: "+name)
        valid = np.isfinite(properties)
        entry["live_shape_coefficients"] = np.where(valid, properties, None).tolist()
        entry["nonfinite_indices"] = np.argwhere(~valid).tolist()
        require(valid.all() and np.array_equal(properties, np.broadcast_to(list(expected.values()), properties.shape)),
                "Actual PhysX surface coefficients differ from inherited [1,1,0]: "+name)
        root = stage.GetPrimAtPath(root_path)
        require(root.IsValid(), "Missing actual surface asset: "+name)
        seen = set()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path = str(prim.GetPath())
            require(path not in seen and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True,
                    "Duplicate/disabled actual surface collider")
            seen.add(path)
            material, rel, resolved = binding(prim)
            entry["bindings"].append(resolved)
            if not material:
                require(not rel, "Dangling or malformed actual material binding must not use scene fallback")
                resolved["fallback_reason"] = "unbound"
            elif not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                # ComputeBoundMaterial('physics') can return an all-purpose visual
                # material. Only this valid, explicitly nonphysics case may fall
                # back; a broken physics-purpose binding must fail closed.
                prim_material = material.GetPrim()
                require(prim_material.IsValid() and prim_material.IsA(UsdShade.Material) and rel
                        and resolved["binding_name"] == "material:binding"
                        and resolved["targets"] == [resolved["resolved_material"]]
                        and not prim_material.HasAPI(PhysxSchema.PhysxMaterialAPI),
                        "Missing physics material API is not a valid all-purpose visual fallback")
                resolved["fallback_reason"] = "nonphysics_all_purpose_visual"
            else:
                require(rel and resolved["targets"] == [resolved["resolved_material"]], "Malformed actual physics surface binding")
                resolved["effective_material"] = read_material(material)
                resolved["resolution"] = "resolved_physics_binding"
                continue
            resolved.update(resolution="PhysicsScene_fallback", effective_material=default_path,
                            fallback_binding=scene_binding["relationship"])
        require(seen, "Missing actual surface CollisionAPI prims: "+name)
        entry["authored_collider_count"] = len(seen)
    audit["verified"] = True
    return audit


def inspect_physics(env, report):
    from regrind.tasks.manager_based.dexterous.mdp.hand2_supported_root import audit_startup
    from regrind.robots.free_hand2_right_hand import inspect_collision_filters
    import omni.usd
    from isaacsim.core.version import get_version
    robot, obj = env.scene["robot"], env.scene["object"]
    audit_startup(robot, env)  # Actual 20 drives/limits/armature/masses/gravity.
    body = {}
    for name, asset in (("hand", robot), ("object", obj)):
        view = asset.root_physx_view
        values = {"mass_kg": array(view.get_masses()), "COM_local_xyzw": array(view.get_coms()),
                  "disable_gravity": array(view.get_disable_gravities())}
        finite(values, name+" live physics")
        require(np.all(values["mass_kg"] > 0) and np.count_nonzero(values["disable_gravity"]) == 0, "Positive mass and gravity ON required")
        body[name] = {"body_names": list(asset.body_names), **{k: v.tolist() for k, v in values.items()}}
    view = obj.root_physx_view
    friction = array(view.get_dof_friction_coefficients()) if int(get_version()[2]) < 5 else array(view.get_dof_friction_properties())[..., 0]
    limits = array(view.get_dof_limits())
    require(np.allclose(limits, [[[0., 2.]]], atol=1e-6, rtol=0) and np.allclose(friction, [[.2]], atol=1e-7, rtol=0), "Actual Scissors 2rad limit/joint friction .2")
    for getter in ("get_dof_stiffnesses", "get_dof_dampings"):
        require(np.count_nonzero(array(getattr(view, getter)())) == 0, "Scissors joint must be passive")
    stage = omni.usd.get_context().get_stage()
    report["actual_body_physics"] = body
    report["actual_hand_drives"] = {name: array(getattr(robot.root_physx_view, getter)()).tolist() for name, getter in
        {"kp": "get_dof_stiffnesses", "kv": "get_dof_dampings", "effort": "get_dof_max_forces",
         "velocity": "get_dof_max_velocities", "armature": "get_dof_armatures", "limits": "get_dof_limits"}.items()}
    report["actual_hand_joint_names"] = list(robot.joint_names)
    report["hand_collision_filters"] = inspect_collision_filters(stage, "/World/envs/env_0/Robot")
    report["actual_object_joint"] = {"names": list(obj.joint_names), "limits_rad": limits.tolist(), "friction": friction.tolist()}
    gravity = np.asarray(env.sim.physics_sim_view.get_gravity())
    require(np.allclose(gravity, [0., 0., -9.81], atol=1e-6, rtol=0), "Actual scene gravity")
    report["actual_gravity_m_s2"] = gravity.tolist()
    table = env.scene["table"]
    raw = array(table.root_physx_view.get_transforms())[0]
    expected_center = [0., 0., .952-env.cfg.scene.table.spawn.size[2]/2]
    require(np.allclose(raw[:3], expected_center, atol=1e-6, rtol=0), "Actual table center/top differs")
    report["actual_table"] = {"transform_xyzw": raw.tolist(), "configured_size_m": list(env.cfg.scene.table.spawn.size),
                              "top_m": float(raw[2]+env.cfg.scene.table.spawn.size[2]/2)}
    report["actual_object_colliders"] = inspect_colliders(stage)
    report["actual_USD_collider_count"] = len(report["actual_object_colliders"])
    inspect_surface_materials(env, stage, report)
    with (C.bundle_path(REPO / "data") / "proof/run.json").open() as f:
        proof = json.load(f)
    report["drake_vs_isaac"] = {"tabletop_m": {"Drake": .945, "Isaac": .952}, **proof["physical_differences"],
                                "USD_collider_count_is_not_Drake_hull_count": True}


class Video:
    def __init__(self, path):
        self.path, self.frames, self.encoder = C.safe_path(path), 0, None
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        st = os.fstat(self.fd)
        self.identity = (st.st_dev, st.st_ino)

    def check_identity(self):
        st = C.regular_file(self.path).stat()
        require((st.st_dev, st.st_ino) == self.identity, "Exclusive video reservation replaced")

    def start(self, sim):
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom, Gf
        from PIL import ImageFont
        self.check_identity()
        camera_path = "/World/ScissorsAuditCamera"
        camera = UsdGeom.Camera.Define(omni.usd.get_context().get_stage(), camera_path)
        camera.CreateFocalLengthAttr().Set(28.0)
        camera.CreateHorizontalApertureAttr().Set(36.0)
        camera.CreateVerticalApertureAttr().Set(20.25)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(.01, 100.))
        sim.set_camera_view((.5, -.7, 1.5), (0., 0., 1.07), camera_prim_path=camera_path)
        self.product = rep.create.render_product(camera_path, (1280, 720))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.annotator.attach([self.product])
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        for _ in range(4):
            sim.render()  # No integration or saved startup frames.
        self.encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "1280x720", "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", f"pipe:{self.fd}"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, pass_fds=(self.fd,))

    def capture(self, sim, row, resets):
        from PIL import Image, ImageDraw
        self.check_identity()
        sim.render()
        rgb = np.asarray(self.annotator.get_data())
        require(rgb.shape in ((720, 1280, 3), (720, 1280, 4)) and rgb.dtype == np.uint8 and rgb[..., :3].any(), "Invalid actual RGB")
        image = Image.fromarray(rgb[..., :3])
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1279, 78), fill="black")
        draw.text((12, 4), "Hand2 Scissors | ZERO RESIDUAL | NOT POLICY", font=self.font, fill="white")
        draw.text((12, 40), f"Control {self.frames+1}/270 | pre-reset reference {row['reference_timestep']}/269 | resets {resets}", font=self.font, fill="white")
        self.encoder.stdin.write(np.asarray(image).tobytes())
        self.frames += 1

    def close(self):
        try:
            if self.encoder is not None:
                encoder, self.encoder = self.encoder, None
                encoder.stdin.close()
                try:
                    rc = encoder.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
                    raise
                error = encoder.stderr.read().decode(errors="replace")
                encoder.stderr.close()
                require(rc == 0, "Video encoder failed: "+error)
            self.check_identity()
            os.fsync(self.fd)
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


def runtime(cache, video, report, app, live):
    import torch
    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry
    from regrind.tasks.manager_based.dexterous.mdp.hand2_supported_root import guarded_native_pd
    from regrind.tasks.manager_based.dexterous.config.wujihand2_scissors import hand2_scissors_env_cfg
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    require(type(cfg) is hand2_scissors_env_cfg.Hand2ScissorsOfficialSEnvCfg, "Must run the actual new Scissors task")
    env = gym.make(TASK, cfg=cfg).unwrapped
    live["env"] = env  # Keep alive even on failure: no destructor before durable evidence.
    # Never env.close(): evidence must be durable before SimulationApp teardown.
    require(env.num_envs == 1 and env.action_manager.total_action_dim == 26 and cfg.decimation == 4 and cfg.sim.dt == 1/120, "Single env/26 actions/120Hz physics")
    robot, obj = env.scene["robot"], env.scene["object"]
    motion = env.command_manager.get_term("motion")
    state = StateWriteAudit()
    state.install_asset(robot, "hand")
    state.install_asset(obj, "object")
    state.install_reset(env, motion)
    wrench = WrenchAudit(env, state, guarded_native_pd)
    try:
        wrench.install()
        inspect_physics(env, report)
        startup_fk(env, C.bundle_path(REPO / "data"), C.load_model_contract(), report)
        with state.section("ordinary_reset_request"):
            initial_observation, _ = env.reset()  # Full ordinary reset, never startup poses in video.
        finite(initial_observation, "initial observations")
        require(int(motion.timesteps_in_current_demo[0]) == 0 and int(motion.current_demo_length[0]) == CONTROLS
                and motion.motion.demo_qpos.shape == (CONTROLS, 34), "Actual 270-control/20-finger Euler qpos34 loader")
        report["loaded_controller_qpos_shape"] = list(motion.motion.demo_qpos.shape)
        report["initial_ordinary_reset_completed"] = True
        install_termination_capture(env, report["controls"], lambda: sample_control(env))
        video.start(env.sim)
        actions = torch.zeros((1, 26), device=env.device)
        start = time.monotonic()
        with torch.inference_mode(), state.section("rollout"):
            for i in range(CONTROLS):
                state.control = i+1
                require(time.monotonic()-start <= TIMEOUT, "Rollout timeout is NOT completion")
                require(app.is_running() and env.sim.is_playing(), "Simulation stopped before full audit")
                observation, reward, terminated, truncated, _ = env.step(actions)
                report["successful_controls"] += 1
                finite({"observation": observation, "reward": reward, "actions": actions}, "step outputs")
                require(len(report["controls"]) == i+1 and len(wrench.physics_steps) == 4*(i+1), "Missing/extra callable or physics sample")
                row = report["controls"][-1]
                record_step_result(row, terminated, truncated)
                row["reward"] = array(reward).tolist()
                require(row["causes"]["hit_table"] == any(p["norm_N"] > 10. for p in row["pads"].values()), "Actual normal hit_table force threshold mismatch")
                report["reset_count"] += int(row["computed_done"])
                zero_object_buffers(obj)
                video.capture(env.sim, row, report["reset_count"])
        require(len(wrench.root_rows) == len(wrench.root_submissions) == len(wrench.physics_steps) == 1080
                and video.frames == CONTROLS, "Incomplete root/control/video counts")
        require(len(state.reset_calls) == len(state.resample_calls) == report["reset_count"]+1, "Ordinary reset accounting mismatch")
    finally:
        report["state_write_audit"] = state.summary()
        report["wrench_audit"] = wrench.summary()
        phases = [r["reference_timestep"] for r in report["controls"] if "reference_timestep" in r]
        report["actual_phase_coverage"] = {"visited_control_indices": sorted(set(phases)), "unique_indices": len(set(phases)),
                                           "full_270_unique_phases_claimed": len(set(phases)) == CONTROLS}
        report["finger_effort_note"] = "applied_torque is an implicit-drive PD estimate, not measured solver torque; root support does not compensate finger joint gravity"


def finalize(report, video, record, snapshot, app):
    """Persist even failed attempts before potentially destructive app.close."""
    try:
        if video is not None:
            video.close()
            report["video_frames"] = video.frames
            report["video_writer_closed"] = True
    except BaseException:
        report["status"] = "failed"
        report["video_finalization_error"] = traceback.format_exc()
    try:
        if video is not None and hasattr(video, "path"):
            # Preserve the partial-file digest even if the encoder failed.
            report["video_sha256"] = C.sha256(video.path)
            if video.frames:
                result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                    "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(video.path)],
                    capture_output=True, text=True, check=True, timeout=60)
                stream, = json.loads(result.stdout)["streams"]
                report["decoded_video"] = stream
                require((stream["width"], stream["height"], stream["r_frame_rate"], int(stream["nb_read_frames"]))
                        == (1280, 720, "30/1", video.frames), "Actual encoded video dimensions/rate/frame count mismatch")
    except BaseException:
        report["status"] = "failed"
        report["video_verification_error"] = traceback.format_exc()
    try:
        report["source_hashes_after"] = verify_snapshot(snapshot)
    except BaseException:
        report["status"] = "failed"
        report["source_finalization_error"] = traceback.format_exc()
    # If this fails, do not close the application first and lose the evidence.
    flush_record(record, report)
    print(json.dumps({"status": report["status"], "successful_controls": report["successful_controls"]}), flush=True)
    code = 0 if report["status"] == "completed_not_physics_certified" else 1
    if app is not None:
        app.close(skip_cleanup=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.run and any(v is None for v in (args.record, args.output_video, args.cache_dir)):
        parser.error("--run requires --record, --output-video and caller-owned --cache-dir")
    if not args.run and any(v is not None for v in (args.record, args.output_video, args.cache_dir)):
        parser.error("Read-only checks take no output/cache arguments")
    checked = precheck()
    if not args.run:
        print(json.dumps({k: v for k, v in checked.items() if k != "source_hashes_before"}, indent=2))
        return 0
    output, record, cache = validate_outputs(args.output_video, args.record, args.cache_dir)
    C.validate_bundle(C.bundle_path(REPO / "data"))
    report = {**checked, "status": "failed", "policy": "NONE", "task": TASK, "physics_certified": False,
              "successful_controls": 0, "reset_count": 0, "controls": [], "video": str(output),
              "video_sampling": "270 post-step/post-normal-reset frames, 30fps 1280x720; no startup frames",
              "physics_assumptions": C.physics_assumptions(), "caller_owned_cache": str(cache),
              "normal_early_resets_allowed": True, "score_gate": False, "functional_success_claimed": False}
    app, video = None, None
    live = {}
    handlers = {}
    def interrupted(signum, frame):
        raise TimeoutError(f"Interrupted signal {signum}; never completed")
    try:
        video = Video(output)
        for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
        signal.alarm(TIMEOUT)
        verify_snapshot(checked["source_hashes_before"])
        os.environ["HAND2_NUM_ENVS"] = "1"
        os.environ["HAND2_USD_CACHE_DIR"] = str(cache / "hand_usd")
        os.environ["REGRIND_DATA_DIR"] = str(REPO / "data")
        from isaaclab.app import AppLauncher  # Never reached by default/--check.
        app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
        runtime(cache, video, report, app, live)
        report["status"] = "completed_not_physics_certified"
    except BaseException:
        report["error"] = traceback.format_exc()
    finally:
        signal.alarm(0)
        # Keep interruption handlers through durable recording; never auto-retry.
    try:
        return finalize(report, video, record, checked["source_hashes_before"], app)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
