#!/usr/bin/env python3
"""Bounded, isolated Hand2 RIGHT Isaac FK validation; never training approval.

Example (in an existing Isaac environment): python -B validate_hand2_right_isaac.py
  --usd-dir /tmp/hand2-unique-cache --output /tmp/hand2-unique-report.json
No dynamics/contact calibration is performed. Missing fixed PhysX frames fail.
Opt-in --shutdown-mode immediate skips graceful cleanup for this standalone app;
its durable report does not certify process exit (the caller must observe that).
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
ROBOTS = REPO / "source/regrind/regrind/robots"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load_module("hand2_right_contract", ROBOTS / "hand2_right_contract.py")


def joint_indices(robot):
    ids, names = robot.find_joints(list(contract.JOINT_NAMES), preserve_order=True)
    if (list(names) != list(contract.JOINT_NAMES) or len(ids) != 20 or len(set(ids)) != 20
            or set(ids) != set(range(20)) or len(robot.joint_names) != 20):
        raise ValueError(f"Non-bijective canonical joint mapping: {ids}, {names}")
    return ids


def check_applied_parameters(robot, ids):
    """Read actual PhysX tensors as well as public articulation limit buffers."""
    view = robot.root_physx_view
    getters = {"kp": "get_dof_stiffnesses", "kv": "get_dof_dampings",
               "effort": "get_dof_max_forces", "velocity": "get_dof_max_velocities",
               "armature": "get_dof_armatures", "limits": "get_dof_limits"}
    result = {}
    for field, method in getters.items():
        if not hasattr(view, method):
            raise RuntimeError(f"Required applied PhysX tensor unavailable: {method}")
        tensor = getattr(view, method)()
        if tuple(tensor.shape) != ((1, 20, 2) if field == "limits" else (1, 20)):
            raise ValueError(f"Unexpected applied tensor shape: {method}: {tensor.shape}")
        values = tensor[0].detach().cpu().tolist()
        canonical = [values[i] for i in ids]
        for name, value in zip(contract.JOINT_NAMES, canonical):
            p = contract.PARAMETERS[name]
            actual = value if field == "limits" else [value]
            expected = [p["lower"], p["upper"]] if field == "limits" else [p[field]]
            if any(not math.isfinite(a) or not math.isclose(a, b, rel_tol=2e-6, abs_tol=1e-8)
                   for a, b in zip(actual, expected)):
                raise ValueError(f"Applied PhysX parameter mismatch: {name}/{field}: {actual} != {expected}")
        result[field] = canonical
    for field, public in (("limits", "joint_pos_limits"), ("velocity", "joint_vel_limits"),
                          ("effort", "joint_effort_limits")):
        if not hasattr(robot.data, public):
            raise RuntimeError(f"Required articulation buffer unavailable: {public}")
        values = getattr(robot.data, public)[0].detach().cpu().tolist()
        if [values[i] for i in ids] != result[field]:
            raise ValueError(f"Articulation buffer differs from applied PhysX tensor: {public}")
    return result


def run_runtime(sim, robot, fixture, torch, stage, model_cfg):
    sim.reset()  # The custom spawn wrapper already authored filters before reset.
    if robot.is_fixed_base or robot.num_instances != 1:
        raise ValueError("Expected one native floating r_mount articulation")
    ids = joint_indices(robot)
    links = fixture["link_names"]
    if len(robot.body_names) != 33 or set(robot.body_names) != set(links):
        raise RuntimeError("PhysX did not preserve all 33 link frames (including massless/fixed tips); "
                           f"actual={robot.body_names}. USD-only transforms cannot replace this check")
    body_ids = {name: robot.body_names.index(name) for name in links}
    masses = model_cfg.check_runtime_masses(robot)  # Must precede gains and hard-set FK.
    report = {"runtime_masses": masses,
              "joint_indices": dict(zip(contract.JOINT_NAMES, ids)),
              "applied_parameters": check_applied_parameters(robot, ids),
              "collision_filters": model_cfg.inspect_collision_filters(stage, "/World/Robot"),
              "poses": []}
    max_position = max_rotation = 0.0
    for pose in fixture["poses"]:
        root = torch.tensor([pose["root_pose_w"]], dtype=torch.float32, device=sim.device)
        joints = torch.tensor([pose["q"][7:]], dtype=torch.float32, device=sim.device)
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), joint_ids=ids)
        sim.forward()  # Kinematics only; no sim.step(), policy, controller or dynamic rollout.
        # Advancing buffer timestamps forces fresh PhysX reads, not a physics step.
        robot.update(sim.get_physics_dt())
        if not hasattr(robot.data, "body_link_pose_w"):
            raise RuntimeError("body_link_pose_w required; COM state is not an FK substitute")
        actual = robot.data.body_link_pose_w[0].detach().cpu().tolist()
        positions = robot.data.joint_pos[0].detach().cpu().tolist()
        for i, expected in zip(ids, pose["q"][7:]):
            if not math.isfinite(positions[i]) or abs(positions[i] - expected) > 1e-6:
                raise ValueError("Joint state did not refresh after hard-set FK")
        errors = [contract.check_pose(actual[body_ids[name]], pose["links"][name]) for name in links]
        errors += [contract.check_pose(actual[body_ids[name]], pose["mapping"][key])
                   for key, name in contract.C.MANO_TO_ROBOT_MAPPING.items()]
        # Read the root from PhysX, rather than accepting the write-through root buffer.
        raw_root = robot.root_physx_view.get_root_transforms()[0].detach().cpu().tolist()
        root_wxyz = raw_root[:3] + [raw_root[6]] + raw_root[3:6]
        errors.append(contract.check_pose(root_wxyz, pose["root_pose_w"]))
        errors.append(contract.check_pose(actual[body_ids["r_wrist"]], pose["wrist_pose_w"]))
        position, rotation = max(p for p, _ in errors), max(r for _, r in errors)
        max_position, max_rotation = max(max_position, position), max(max_rotation, rotation)
        report["poses"].append({"configuration": pose["configuration"], "root_variant": pose["root_variant"],
                                "max_position_error_m": position, "max_rotation_error_rad": rotation,
                                "links_checked": 33, "mapping_frames_checked": 21})
    report["max_position_error_m"] = max_position
    report["max_rotation_error_rad"] = max_rotation
    report["collision_filters"] = model_cfg.inspect_collision_filters(stage, "/World/Robot")
    return report


def save_report(stream, report):
    """Persist through the exclusively reserved descriptor before Isaac shutdown."""
    stream.seek(0)
    stream.truncate()
    json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=REPO / "data/validation/hand2_right_fk_v1.json")
    parser.add_argument("--usd-dir", required=True, help="Caller-owned, never-used USD cache directory")
    parser.add_argument("--output", required=True, help="New JSON report; no overwrite or symlinks")
    parser.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
    parser.add_argument("--shutdown-mode", choices=("graceful", "immediate"), default="graceful",
                        help="Application shutdown (default: graceful); immediate skips graceful cleanup")
    args = parser.parse_args(argv)
    immediate = args.shutdown_mode == "immediate"
    # All stdlib contract/argument failures precede AppLauncher import and launch.
    fixture_path = contract.safe_path(args.fixture)
    fixture = contract.validate_fixture(json.loads(fixture_path.read_text()))
    usd_dir, output = contract.new_path(args.usd_dir), contract.new_path(args.output)
    if usd_dir == output or usd_dir.is_relative_to(output) or output.is_relative_to(usd_dir):
        raise ValueError("Cache and report paths must be separate")
    report = {"status": "failed", "validation_status": "failed",
              "shutdown_mode": args.shutdown_mode, "immediate_requested": immediate,
              "validation_only": True, "approved_for_training": False,
              "parameter_status": contract.PARAMETER_STATUS,
              "fixture_sha256": contract.C.sha256(fixture_path), "provenance": fixture["provenance"],
              "runtime_executed": False, "dynamic_steps_requested": 0,
              "runtime_mass_verification": "not_verified",
              "contact_or_dynamics_verified": False, "headless": True, "enable_cameras": False,
              "configuration": "single floating Hand2 RIGHT, gravity disabled, non-instanceable validation USD"}
    # Reserve the report exclusively before launch; exceptions cannot overwrite another run.
    stream = output.open("x")
    app = sim = None
    try:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True, enable_cameras=False, device=args.device)
        app = launcher.app
        report["runtime_executed"] = True
        # Runtime imports MUST follow application creation.
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        import omni.usd
        from pxr import UsdPhysics

        model_cfg = load_module("free_hand2_right_hand", ROBOTS / "free_hand2_right_hand.py")
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device, use_fabric=True))
        cfg = model_cfg.make_hand2_right_cfg("/World/Robot", usd_dir)
        robot = Articulation(cfg)
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath("/World/Robot").IsValid():
            raise RuntimeError("Robot spawn failed")
        # Keep the imported physics schema accessible only after AppLauncher.
        if UsdPhysics is None:
            raise RuntimeError("USD physics schema unavailable")
        report["runtime"] = run_runtime(sim, robot, fixture, torch, stage, model_cfg)
        report["runtime_mass_verification"] = report["runtime"].get("runtime_masses", {}).get(
            "verification_state", "not_verified")
        report["validation_status"] = "pass"
    except (Exception, SystemExit) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        with stream:
            report["application_close"] = "pending" if app is not None else "not_created"
            report["cleanup_status"] = ("not_created" if sim is None else
                                        "skipped_immediate" if immediate else "pending")
            if immediate and report["validation_status"] == "pass":
                report["status"] = "validation_pass_shutdown_requested"
            save_report(stream, report)
            if sim is not None and not immediate:
                # IsaacLab's headless context-manager teardown, only for our sim.
                # A failed phase must not prevent the remaining public cleanup calls.
                for phase in ("stop", "clear_all_callbacks", "clear_instance"):
                    report["cleanup_status"] = phase
                    save_report(stream, report)
                    try:
                        getattr(sim, phase)()
                    except (Exception, SystemExit) as error:
                        report.setdefault("cleanup_errors", []).append({
                            "phase": phase, "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc()})
                        report["cleanup_status"] = "failed"
                        save_report(stream, report)
                report["cleanup_status"] = "failed" if report.get("cleanup_errors") else "passed"
                if report["validation_status"] == "pass" and report["cleanup_status"] == "passed":
                    report["status"] = "pass"
                save_report(stream, report)  # Durable final cleanup result before app.close().
            if app is not None:
                try:
                    # Only the application created by this invocation; no context cleanup
                    # in immediate mode. Use SimulationApp's documented public option.
                    if immediate:
                        app.close(skip_cleanup=True)
                    else:
                        app.close()
                except Exception as error:
                    report["status"] = "failed"
                    report["application_close"] = "failed"
                    report["close_error"] = f"{type(error).__name__}: {error}"
                    report["close_traceback"] = traceback.format_exc()
                else:
                    report["application_close"] = "closed"
                    if immediate and report["validation_status"] == "pass":
                        report["status"] = "validation_pass_shutdown_complete"
                # SystemExit (or process termination) leaves the durable pending report.
                save_report(stream, report)
    print(f"Hand2 validation status={report['status']}; report={output}; training approval=false")
    return 0 if report["status"] in ("pass", "validation_pass_shutdown_complete") else 1


if __name__ == "__main__":
    sys.exit(main())
