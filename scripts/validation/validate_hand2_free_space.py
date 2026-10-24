#!/usr/bin/env python3
"""Local Hand2 free-space A/B tracking probe, not controller calibration.

A: gravity OFF. B: gravity ON + ideal root-only net weight support, not arm
dynamics. Both use identical vendor finger drives and root PD, 120 Hz physics,
30 Hz held targets, and simultaneous labelled 497-frame video. No scene objects.
Run only in the existing Isaac environment, with a NEW --output-dir.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import traceback

REPO = Path(__file__).resolve().parents[2]
ROBOTS = REPO / "source/regrind/regrind/robots"
REFERENCE = REPO / "data/hand2_right/bowl/retargeted_120fps.h5"
REFERENCE_SHA256 = "cf2651b843c620e25c1aad1776b3f07f4fc930bac0a2f52c70a19ce30cb4a357"
PHYSICS_HZ, CONTROL_HZ, DECIMATION = 120, 30, 4
REFERENCE_FRAMES, CONTROL_FRAMES = 1985, 497
GRAVITY = (0.0, 0.0, -9.81)
OFFSETS = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
LABELS = ("A: gravity OFF", "B: gravity ON + root weight support")
# Pinned offline Drake CPU calculation; never recomputed in the Isaac runtime.
ROOT_INERTIA_DERIVATION = {
    "source_urdf_sha256": "9ad34330b2fd30b4d4d192ecd4d8a52e80fa266c938d0d51e9fe454267c42d86",
    "canonical_urdf_sha256": "932e8f4c052c26ccc70b93756f8a888c42e41a576ea75de30be4db417d9d6a7f",
    "reference_sha256": REFERENCE_SHA256,
    "method": "Drake mass matrix M, root angular velocities first: S = M[:3,:3] - M[:3,3:] @ solve(M[3:,3:], M[3:,:3]); eigvalsh(S)",
    "sampling": "497 reference poses at native indices 4*k, k=0..496; canonical URDF, zero armature",
    "extrema": "I_min/I_max are min/max over all three eigenvalues at all 497 sampled poses",
    "min_eigenvalue_kg_m2": 0.0001825096675712572,
    "max_eigenvalue_kg_m2": 0.0008118712061139207,
    "natural_frequency_rad_s": 10.0, "damping_ratio": 1.0,
    "kp_formula": "I_min * wn**2", "kd_formula": "2 * zeta * I_min * wn",
    "disclaimer": "conservative zero-armature Schur heuristic, not stability proof",
}
ROOT_PROFILES = {
    "original": (3.0, 0.3),
    "inertia-derived": (0.01825096675712572, 0.0036501933514251437),
    "inertia-tracking": (ROOT_INERTIA_DERIVATION["min_eigenvalue_kg_m2"] * 625,
                         ROOT_INERTIA_DERIVATION["min_eigenvalue_kg_m2"] * 50),
}
PD_FORCE_ABORT_N, PD_TORQUE_ABORT_NM = 50.0, 1.0
REFERENCE_VELOCITY_METHOD = (
    "497 native reference[::4] poses, dt=1/30 s; position: np.gradient(axis=0, edge_order=1); "
    "WORLD omega: Log(R[k+1] @ R[k-1].T)/(2*dt), final Log(R[-1] @ R[-2].T)/dt; "
    "first linear/angular velocities zero; wxyz converted to SciPy xyzw; no filtering"
)


def root_controller(root_profile="original"):
    if root_profile not in ROOT_PROFILES:
        raise ValueError(f"Unknown root profile: {root_profile}")
    kp, kd = ROOT_PROFILES[root_profile]
    label = f"{root_profile} candidate" if root_profile != "original" else "original"
    result = {"profile": root_profile, "label": f"angular profile: {label}, NOT calibrated",
              "kp_pos": 300.0, "kd_pos": 30.0, "kp_rot": kp, "kd_rot": kd,
              "root_wrench_cap": None, "velocity_target": 0.0,
              "state_provenance": {"source": "robot.data.root_state_w",
                                   "pose": "root LINK origin/orientation in WORLD",
                                   "linear_velocity": "root body COM in WORLD (legacy profile)",
                                   "angular_velocity": "root body angular velocity in WORLD"},
              "pd_abort_limits": {"force_N": PD_FORCE_ABORT_N, "torque_Nm": PD_TORQUE_ABORT_NM,
                                  "scope": "PD only, excludes B support; abort, no clipping or fallback"}}
    if root_profile in ("inertia-derived", "inertia-tracking"):
        result["derivation"] = dict(ROOT_INERTIA_DERIVATION)
        result["derivation"]["scalar_kd_dt_over_I_min"] = kd / PHYSICS_HZ / ROOT_INERTIA_DERIVATION["min_eigenvalue_kg_m2"]
    if root_profile == "inertia-tracking":
        result["derivation"]["natural_frequency_rad_s"] = 25.0
        result["velocity_target"] = {"scope": "root LINK origin linear / WORLD angular velocity, same for A/B",
                                     "method": REFERENCE_VELOCITY_METHOD}
        result["state_provenance"]["source"] = "robot.data.root_link_state_w"
        result["state_provenance"]["linear_velocity"] = "root LINK origin in WORLD, matching reference derivative"
        result["default_profile"] = "original"
        result["acceleration_feedforward"] = False
        result["finger_feedforward"] = False
    return result


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_array(value, shape=None):
    import numpy as np
    result = np.asarray(value, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError(f"Nonfinite or malformed array; expected {shape}, got {result.shape}")
    return result


def root_support_wrench(body_masses, body_com_w, root_application_point_w, gravity):
    """Cancel net gravity wrench about the explicit root-link application point."""
    import numpy as np
    masses = finite_array(body_masses)
    if masses.ndim != 1 or not len(masses) or np.any(masses <= 0):
        raise ValueError("Positive finite body masses required")
    com = finite_array(body_com_w, (len(masses), 3))
    root = finite_array(root_application_point_w, (3,))
    g = finite_array(gravity, (3,))
    weight = masses[:, None] * g
    return -weight.sum(axis=0), -np.cross(com - root, weight).sum(axis=0)


def rotation_error(target_quat, actual_quat):
    """World axis-angle of target * conjugate(actual), wxyz and shortest arc."""
    import numpy as np
    target, actual = (finite_array(q, (4,)) for q in (target_quat, actual_quat))
    for q in (target, actual):
        if abs(np.linalg.norm(q) - 1.0) > 1e-5:
            raise ValueError("Unit wxyz quaternion required")
    target, actual = target / np.linalg.norm(target), actual / np.linalg.norm(actual)
    w = target[0] * actual[0] + np.dot(target[1:], actual[1:])
    xyz = -target[0] * actual[1:] + actual[0] * target[1:] - np.cross(target[1:], actual[1:])
    # Tie-break exact pi as well, so +/- quaternion representations agree.
    if w < 0 or (w == 0 and xyz[np.argmax(np.abs(xyz))] < 0):
        w, xyz = -w, -xyz
    sine = np.linalg.norm(xyz)
    return xyz * (2.0 if sine < 1e-10 else 2.0 * np.arctan2(sine, w) / sine)


def root_controller_state(robot, root_profile="original"):
    """Tracking FF needs LINK velocity; preserve actual COM velocity for old profiles."""
    if root_profile not in ROOT_PROFILES:
        raise ValueError(f"Unknown root profile: {root_profile}")
    return robot.data.root_link_state_w[0] if root_profile == "inertia-tracking" else robot.data.root_state_w[0]


def root_pd_wrench(target_pose, root_state, root_profile="original", reference_velocity=None):
    """Root PD: tracking requires WORLD LINK velocity; old profiles retain COM damping."""
    gains = root_controller(root_profile)
    target = finite_array(target_pose, (7,))
    state = finite_array(root_state, (13,))
    velocity = finite_array([0.0] * 6 if reference_velocity is None else reference_velocity, (6,))
    if root_profile == "inertia-tracking":
        force = 300.0 * (target[:3] - state[:3]) + 30.0 * (velocity[:3] - state[7:10])
        torque = gains["kp_rot"] * rotation_error(target[3:], state[3:7]) + gains["kd_rot"] * (velocity[3:] - state[10:13])
    else:
        if (velocity != 0).any():
            raise ValueError("Root velocity feedforward requires inertia-tracking")
        force = 300.0 * (target[:3] - state[:3]) - 30.0 * state[7:10]
        torque = gains["kp_rot"] * rotation_error(target[3:], state[3:7]) - gains["kd_rot"] * state[10:13]
    return force, torque


def pd_wrench_guard(pd):
    """Abort on nonfinite/excessive PD norms before write; never limit B support."""
    import numpy as np
    force, torque = (finite_array(value, (3,)) for value in pd)
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.array([np.linalg.norm(force), np.linalg.norm(torque)])
    if not np.isfinite(norms).all():
        raise ValueError("Nonfinite PD force/torque norm")
    if norms[0] > PD_FORCE_ABORT_N or norms[1] > PD_TORQUE_ABORT_NM:
        raise ValueError(f"PD wrench exceeds abort limits 50 N / 1 Nm: {norms[0]} N / {norms[1]} Nm")
    return force, torque


def tracking_guard(target_pose, root_state, joints, joint_velocities):
    import numpy as np
    target = finite_array(target_pose, (7,))
    state = finite_array(root_state, (13,))
    finite_array(joints, (20,))
    finite_array(joint_velocities, (20,))
    error = float(np.linalg.norm(target[:3] - state[:3]))
    if error > 0.5:
        raise ValueError(f"Root position error exceeds 0.5 m: {error}")
    return error, float(np.linalg.norm(rotation_error(target[3:], state[3:7])))


def control_indices(frame_count=REFERENCE_FRAMES):
    if frame_count != REFERENCE_FRAMES:
        raise ValueError("Expected exactly 1985 native 120 Hz reference frames")
    return tuple(range(0, frame_count, DECIMATION))


def reference_targets(reference, control_index, offset):
    import numpy as np
    if not 0 <= control_index < CONTROL_FRAMES:
        raise ValueError("Control index out of range")
    frame = control_indices()[control_index]
    pose = np.r_[reference["robot_pos"][frame] + finite_array(offset, (3,)),
                 reference["robot_quat"][frame]]
    return pose, reference["robot_joints"][frame].copy()


def reference_velocities(reference):
    """Return finite (497, 6) WORLD root velocities; never modify native targets."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    positions = finite_array(reference["robot_pos"], (REFERENCE_FRAMES, 3))[::DECIMATION]
    quaternions = finite_array(reference["robot_quat"], (REFERENCE_FRAMES, 4))[::DECIMATION]
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Reference has nonunit wxyz quaternions")
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    dt = 1 / CONTROL_HZ
    velocity = np.zeros((CONTROL_FRAMES, 6))
    velocity[:, :3] = np.gradient(positions, dt, axis=0, edge_order=1)
    velocity[1:-1, 3:] = Rotation.from_matrix(
        rotations[2:] @ rotations[:-2].transpose(0, 2, 1)).as_rotvec() / (2 * dt)
    velocity[-1, 3:] = Rotation.from_matrix(rotations[-1] @ rotations[-2].T).as_rotvec() / dt
    velocity[0] = 0.0  # The initial root velocity is hard-set to zero in BOTH cases.
    return finite_array(velocity, (CONTROL_FRAMES, 6))


def tracking_check(cases=None):
    """Predetermined v3 free-space qualification, separate from comparison completion."""
    result = {"version": "v3", "scope": "free-space tracking only, NOT grasp qualification or calibration",
              "evaluated": False, "qualified": False,
              "limits_strict_less_than": {"mean_root_position_m": 0.010,
                                           "mean_root_rotation_rad": 0.1,
                                           "each_finger_joint_rmse_rad": 0.05},
              "cases": []}
    if cases is None:
        return result
    if len(cases) != 2:
        raise ValueError("Tracking qualification requires BOTH A/B cases")
    for case in cases:
        position = finite_array(case["root_position_error_m"]["mean"], ())
        rotation = finite_array(case["root_rotation_error_rad"]["mean"], ())
        joints = finite_array(list(case["joint_rmse_rad"].values()), (20,))
        checks = {"mean_root_position": bool(0 <= position < 0.010),
                  "mean_root_rotation": bool(0 <= rotation < 0.1),
                  "all_finger_joints": bool(((joints >= 0) & (joints < 0.05)).all())}
        result["cases"].append({"label": case["label"], "checks": checks,
                                "qualified": all(checks.values())})
    result["evaluated"] = True
    result["qualified"] = all(case["qualified"] for case in result["cases"])
    return result


def validate_reference(reference, names, contract):
    import numpy as np
    if list(names) != list(contract.JOINT_NAMES):
        raise ValueError("Reference must use exactly the canonical 20 joint names")
    for field, width in (("robot_pos", 3), ("robot_quat", 4), ("robot_joints", 20)):
        reference[field] = finite_array(reference[field], (REFERENCE_FRAMES, width))
    if not np.allclose(np.linalg.norm(reference["robot_quat"], axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Reference has nonunit wxyz quaternions")
    lower = np.array([p["lower"] for p in contract.PARAMETERS.values()])
    upper = np.array([p["upper"] for p in contract.PARAMETERS.values()])
    q = reference["robot_joints"]
    if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6):
        raise ValueError("Reference joint limits violated")
    reference["robot_joints"] = np.clip(q, lower, upper)  # Same target clipping as task.
    return reference


def preflight(contract):
    """Only stdlib and existing sim-free contract, before AppLauncher import."""
    contract.validate_asset()
    manifest_path = REFERENCE.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (sha256(REFERENCE) != REFERENCE_SHA256
            or manifest["output_sha256"][REFERENCE.name] != REFERENCE_SHA256
            or manifest["frames"] != REFERENCE_FRAMES or manifest["control_frames"] != CONTROL_FRAMES
            or manifest["fps"] != PHYSICS_HZ or manifest["root_frame"] != "r_mount"
            or manifest["joint_names"] != list(contract.JOINT_NAMES)):
        raise ValueError("Approved runtime reference hash/layout mismatch")
    return {"reference_sha256": REFERENCE_SHA256, "reference_manifest_sha256": sha256(manifest_path),
            "probe_sha256": sha256(__file__), "factory_sha256": sha256(ROBOTS / "free_hand2_right_hand.py"),
            "parameter_auditor_sha256": sha256(Path(__file__).with_name("validate_hand2_right_isaac.py")),
            "asset": contract.provenance()}


def inspect_scene(stage, paths, expected_names, usd_modules=None):
    """Check all 66 actual rigid-body gravity flags before reset; no other colliders."""
    if usd_modules is None:
        from pxr import Usd, UsdPhysics, PhysxSchema
    else:
        Usd, UsdPhysics, PhysxSchema = usd_modules
    reports = []
    for index, path in enumerate(paths):
        bodies = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                flag = PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()
                if prim.GetName() in bodies or flag is not (index == 0):
                    raise ValueError(f"Wrong/duplicate gravity body: {prim.GetPath()}: {flag}")
                bodies[prim.GetName()] = flag
        if len(bodies) != 33 or set(bodies) != set(expected_names):
            raise ValueError("Gravity audit must cover exactly 33 named links per case")
        reports.append(bodies)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if not any(str(prim.GetPath()).startswith(path + "/") for path in paths):
                raise ValueError(f"Non-hand collision/rigid body in free space: {prim.GetPath()}")
    return reports


def initialize_case(robot, ids, pose, joints, tensor):
    """Only initial hard-set; caller never invokes this within tracking steps."""
    robot.reset()
    robot.write_root_pose_to_sim(tensor(pose).reshape(1, 7))
    robot.write_root_velocity_to_sim(tensor([0.0] * 6).reshape(1, 6))
    q = tensor(joints).reshape(1, 20)
    zero = q * 0.0
    robot.write_joint_state_to_sim(q, zero, joint_ids=ids)
    robot.set_joint_position_target(q, joint_ids=ids)
    robot.set_joint_velocity_target(zero, joint_ids=ids)


def apply_root_wrench(robot, root_id, pd, support, root_link_pos_w, tensor):
    """Exactly one world-frame root write per physics step; support cannot be lost."""
    pd_force, pd_torque = pd_wrench_guard(pd)
    force = pd_force + finite_array(support[0], (3,))
    torque = pd_torque + finite_array(support[1], (3,))
    finite_array(force, (3,))
    finite_array(torque, (3,))
    robot.set_external_force_and_torque(
        forces=tensor(force).reshape(1, 1, 3), torques=tensor(torque).reshape(1, 1, 3),
        positions=tensor(finite_array(root_link_pos_w, (3,))).reshape(1, 1, 3),
        body_ids=[root_id], is_global=True)
    return force, torque


def save_summary(path, report):
    with path.open("w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_probe(output, report, writer, model, root_profile="original"):
    controller = root_controller(root_profile)
    # All simulator/third-party runtime imports occur after AppLauncher in main.
    import h5py
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdGeom
    from PIL import Image, ImageDraw

    auditor = load_module("hand2_free_space_parameter_auditor", Path(__file__).with_name("validate_hand2_right_isaac.py"))
    contract = model.contract
    with h5py.File(REFERENCE) as file:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in file["robot_joint_names"][()]]
        if file.attrs["root_frame"] != "r_mount" or file.attrs["quaternion_order"] != "wxyz" or file.attrs["fps"] != 120:
            raise ValueError("Unexpected native root/quaternion/FPS contract")
        reference = validate_reference({k: file[k][()] for k in ("robot_pos", "robot_quat", "robot_joints")}, names, contract)
    root_velocities = reference_velocities(reference) if root_profile == "inertia-tracking" else None
    sampled = np.array(control_indices())
    speeds = np.abs(np.diff(reference["robot_joints"][sampled], axis=0)) * CONTROL_HZ
    report["target_speed"] = {"joint_max_rad_s": dict(zip(names, speeds.max(axis=0).tolist())),
        "root_max_m_s": float(np.linalg.norm(np.diff(reference["robot_pos"][sampled], axis=0), axis=1).max() * CONTROL_HZ),
        "root_max_rad_s": float(max(np.linalg.norm(rotation_error(a, b)) * CONTROL_HZ for a, b in
                                   zip(reference["robot_quat"][sampled][1:], reference["robot_quat"][sampled][:-1]))),
        "meaning": ("adjacent finite-difference speed diagnostic; root velocity FF uses root_pd.velocity_target.method"
                    if root_profile == "inertia-tracking" else
                    "finite difference of 30 Hz targets; NOT supplied as velocity feedforward")}
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1 / PHYSICS_HZ, render_interval=DECIMATION, device="cuda:0", gravity=GRAVITY, use_fabric=True))
    stage = omni.usd.get_context().get_stage()
    paths = ["/World/CaseA/Robot", "/World/CaseB/Robot"]
    robots = []
    for index, path in enumerate(paths):
        UsdGeom.Xform.Define(stage, path.rsplit("/", 1)[0])
        cfg = model.make_hand2_right_cfg(path, output / f"usd_cache_{index}")
        cfg.spawn.rigid_props.disable_gravity = index == 0
        pose, joints = reference_targets(reference, 0, OFFSETS[index])
        cfg.init_state.pos, cfg.init_state.rot = tuple(pose[:3]), tuple(pose[3:])
        cfg.init_state.joint_pos = dict(zip(names, joints.tolist()))
        robots.append(Articulation(cfg))
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85))
    light.func("/World/Light", light)
    camera_path = "/World/ProbeCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateProjectionAttr().Set("orthographic")
    camera.CreateHorizontalApertureAttr().Set(29.0)  # 2.9 m at USD's tenth-unit aperture convention.
    camera.CreateVerticalApertureAttr().Set(10.875)
    report["gravity_disable_flags_before_reset"] = inspect_scene(stage, paths, contract.model_layout()[0])
    sim.reset()
    # FIRST post-reset action: live PhysX mass guards on BOTH cases, before any physics step.
    report["runtime_masses"] = [model.check_runtime_masses(robot) for robot in robots]
    cases = []
    tensor = lambda v: torch.as_tensor(v, device=sim.device, dtype=torch.float32)
    array = lambda v: v.detach().cpu().numpy()
    for index, robot in enumerate(robots):
        if robot.is_fixed_base or robot.num_instances != 1:
            raise ValueError("Expected independent native floating Hand2 articulations")
        ids = auditor.joint_indices(robot)
        applied = auditor.check_applied_parameters(robot, ids)
        pose, joints = reference_targets(reference, 0, OFFSETS[index])
        initialize_case(robot, ids, pose, joints, tensor)
        case = {"label": LABELS[index], "root_offset_m": list(OFFSETS[index]),
                "vendor_applied_parameters": applied, "physics_steps": 0, "control_frames": 0,
                "estimated_torque_scope": "robot.data.applied_torque: clipped implicit-PD estimate, NOT hardware or measured joint force",
                "root_force_max_N": 0.0, "root_torque_max_Nm": 0.0,
                "estimated_effort_ratio_max": 0.0, "estimated_saturation_count": 0}
        cases.append({"robot": robot, "ids": ids, "root_id": robot.body_names.index("r_mount"),
                      "masses": np.array([report["runtime_masses"][index]["body_mass_kg_per_env"][0][n] for n in robot.body_names]),
                      "report": case, "joint_sq": np.zeros(20), "root_errors": [], "rotation_errors": []})
    report["cases"] = [c["report"] for c in cases]
    minimum_gap = float("inf")
    sim.forward()  # Initial kinematics only; no physics advancement.
    for robot in robots:
        robot.update(1 / PHYSICS_HZ)
    product = rep.create.render_product(camera_path, (1280, 480))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([product])

    def aim(frame):
        center = reference["robot_pos"][frame] + np.array([1.0, 0.0, 0.05])
        sim.set_camera_view(tuple(center + [0.0, -3.0, 1.0]), tuple(center), camera_prim_path=camera_path)

    aim(0)
    for _ in range(4):
        sim.render()  # Warm up rendering only, after the mass checks.
    for control, frame in enumerate(control_indices()):
        targets = [reference_targets(reference, control, offset) for offset in OFFSETS]
        for case, (_, joints) in zip(cases, targets):
            case["robot"].set_joint_position_target(tensor(joints).reshape(1, 20), joint_ids=case["ids"])
        for _ in range(DECIMATION):
            for index, (case, (pose, _)) in enumerate(zip(cases, targets)):
                robot, root_id = case["robot"], case["root_id"]
                state = array(robot.data.root_state_w[0])
                tracking_guard(pose, state, array(robot.data.joint_pos[0, case["ids"]]),
                               array(robot.data.joint_vel[0, case["ids"]]))
                controller_state = array(root_controller_state(robot, root_profile))
                pd = root_pd_wrench(pose, controller_state, root_profile,
                                    reference_velocity=None if root_velocities is None else root_velocities[control])
                coms = finite_array(array(robot.data.body_com_pos_w[0]), (33, 3))
                support = root_support_wrench(case["masses"], coms, state[:3], GRAVITY) if index else (np.zeros(3), np.zeros(3))
                force, torque = apply_root_wrench(robot, root_id, pd, support, state[:3], tensor)
                case["report"]["root_force_max_N"] = max(case["report"]["root_force_max_N"], float(np.linalg.norm(force)))
                case["report"]["root_torque_max_Nm"] = max(case["report"]["root_torque_max_Nm"], float(np.linalg.norm(torque)))
                robot.write_data_to_sim()
            sim.step(render=False)
            for case, (pose, _) in zip(cases, targets):
                robot, record = case["robot"], case["report"]
                robot.update(1 / PHYSICS_HZ)
                tracking_guard(pose, array(robot.data.root_state_w[0]), array(robot.data.joint_pos[0, case["ids"]]),
                               array(robot.data.joint_vel[0, case["ids"]]))
                estimate = finite_array(array(robot.data.applied_torque[0, case["ids"]]), (20,))
                ratio = np.abs(estimate) / np.array(record["vendor_applied_parameters"]["effort"])
                record["estimated_effort_ratio_max"] = max(record["estimated_effort_ratio_max"], float(ratio.max()))
                record["estimated_saturation_count"] += int((ratio >= 0.99).sum())
                record["physics_steps"] += 1
            # A 2 m separation, with bounded root error and measured body envelope;
            # fail rather than claim free space if hands approach each other.
            a, b = [finite_array(array(r.data.body_com_pos_w[0]), (33, 3)) for r in robots]
            gap = float(b[:, 0].min() - a[:, 0].max())
            minimum_gap = min(minimum_gap, gap)
            if gap <= 0.5:
                raise ValueError("Inter-case COM envelope separation below 0.5 m")
        for case, (pose, joints) in zip(cases, targets):
            robot = case["robot"]
            errors = tracking_guard(pose, array(robot.data.root_state_w[0]), array(robot.data.joint_pos[0, case["ids"]]),
                                    array(robot.data.joint_vel[0, case["ids"]]))
            case["root_errors"].append(errors[0])
            case["rotation_errors"].append(errors[1])
            case["joint_sq"] += (array(robot.data.joint_pos[0, case["ids"]]) - joints) ** 2
            case["report"]["control_frames"] += 1
        aim(frame)
        sim.render()
        rgb = np.asarray(annotator.get_data())
        if rgb.shape not in ((480, 1280, 4), (480, 1280, 3)) or rgb.dtype != np.uint8 or not np.any(rgb[:, :, :3]):
            raise ValueError(f"Invalid/empty RGB render: {rgb.shape}")
        image = Image.fromarray(rgb[:, :, :3])
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1280, 49), fill="black")
        for index, label in enumerate(LABELS):
            draw.text((16 + index * 640, 6), label, fill="white")
        draw.text((16, 28), f"SAME TIME {control + 1}/497 | 120 Hz physics / 30 Hz targets | vendor finger PD | {controller['label']}", fill="white")
        writer.append_data(np.asarray(image))  # Stream immediately; never retain video frames.
        report["video_frames"] = control + 1
    for case in cases:
        record = case["report"]
        rmse = np.sqrt(case["joint_sq"] / CONTROL_FRAMES)
        record["joint_rmse_rad"] = dict(zip(names, rmse.tolist()))
        record["all_joint_rmse_rad"] = float(np.sqrt(np.mean(rmse ** 2)))
        record["finger_rmse_rad"] = {finger: float(np.sqrt(np.mean(rmse[i * 4:(i + 1) * 4] ** 2)))
                                      for i, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"))}
        record["root_position_error_m"] = {"mean": float(np.mean(case["root_errors"])), "max": float(max(case["root_errors"]))}
        record["root_rotation_error_rad"] = {"mean": float(np.mean(case["rotation_errors"])), "max": float(max(case["rotation_errors"]))}
        record["estimated_effort_saturation_fraction"] = record["estimated_saturation_count"] / (record["physics_steps"] * 20)
    if root_profile == "inertia-tracking":
        report["tracking_check"] = tracking_check(report["cases"])
    report["measurement_timing"] = "root/joint lag: after four physics steps against same held target 4*k; estimated torque: each physics step"
    report["inter_case_interaction"] = {"target_separation_m": 2.0, "minimum_COM_envelope_x_gap_m": minimum_gap,
                                        "scope": "geometric separation check each step; no cross-contact sensor measurement"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New LOCAL directory, parent must exist; never overwrite")
    parser.add_argument("--root-profile", choices=tuple(ROOT_PROFILES), default="original",
                        help="Local root PD profile for BOTH cases; inertia-derived/inertia-tracking candidate, NOT calibrated; default: original")
    args = parser.parse_args(argv)  # --help before numpy, torch, Isaac, or application startup.
    model = load_module("hand2_free_space_factory", ROBOTS / "free_hand2_right_hand.py")
    hashes = preflight(model.contract)
    output = model.contract.new_path(args.output_dir)
    output.mkdir()
    summary, video = output / "summary.json", output / "tracking.mp4"
    report = {"status": "failed", "runtime_executed": False, "controller_calibrated": False,
              "approved_physics": False, "approved_for_training": False, "source_hashes": hashes,
              "expected_mass_contract": model.expected_mass_contract(), "video_frames": 0,
              "root_pd": root_controller(args.root_profile),
              "support_model": "B only: -sum(m)*g, -sum((COM_i_W-root_LINK_origin_W) cross m_i*g), explicitly applied at root link origin; ideal net weight support, NOT arm dynamics",
              "finger_feedforward": False, "per_body_gravity_compensation": False,
              "scene": "two independent native Hand2; dome light/camera only, no Bowl/table/Ground",
              "video": {"path": str(video), "fps": 30, "resolution": [1280, 480], "layout": "simultaneous labelled A/B"},
              "physics_hz": PHYSICS_HZ, "control_hz": CONTROL_HZ, "decimation": DECIMATION,
              "shutdown": "own app.close(skip_cleanup=True); fsync video/summary/directory before shutdown"}
    if args.root_profile == "inertia-tracking":
        report["tracking_check"] = tracking_check()
    app = writer = None
    status = 1
    try:
        save_summary(summary, report)
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
        report["runtime_executed"] = True
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(video), fps=CONTROL_HZ, codec="libx264", macro_block_size=16)
        run_probe(output, report, writer, model, args.root_profile)
        writer.close()
        writer = None
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "null", "-"],
                       check=True, timeout=60, capture_output=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height", "-of", "json", str(video)],
                               check=True, timeout=30, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        if (int(stream["nb_read_frames"]) != CONTROL_FRAMES or stream["avg_frame_rate"] != "30/1"
                or (stream["width"], stream["height"]) != (1280, 480)):
            raise ValueError("Decoded video count/FPS/resolution mismatch")
        report["video"]["decoded"] = stream
        report["video"]["sha256"] = sha256(video)
        report["status"] = "comparison_complete_pending_review_not_calibrated"
        status = 0
    except (Exception, SystemExit) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as error:
                report["video_close_error"] = str(error)
        try:
            if video.exists():
                with video.open("rb") as stream:
                    os.fsync(stream.fileno())
            report["application_close"] = "pending" if app is not None else "not_created"
            save_summary(summary, report)
            fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            # Never stop a shared app; this invocation owns its sole AppLauncher.
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
                # A terminating close intentionally leaves the durable pending state.
                save_summary(summary, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
