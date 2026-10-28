"""Portable Scissors contract, before AppLauncher; no Drake or source workspace.

Pinned Stage A evidence is geometry evidence, not a physics certificate. Historical
false flags are never promoted. Receipt structure cannot authenticate user consent.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import tempfile

import h5py
import numpy as np

FRAMES, NATIVE_FRAMES, CONTROLS, FPS = 641, 161, 270, 120
SPEED = 160 / 269
WINDOW = [25, 185]
NAMESPACE = "hand2_right/scissors_official_v1"
SOURCE = "data/arctic/retargeted/hand2_right/scissors_official_feasible_v1"
ORIGINAL = "data/arctic/retargeted/hand2_right/scissors_official_v1"
REPO = "repos/regrind/"
ASSETS = REPO + "source/regrind/regrind/assets/scissors/"
DEMO = REPO + "data/arctic_demo/arctic_leap_scissors/demo_interp_120fps.h5"
NATIVE_DEMO = REPO + "data/arctic_demo/arctic_leap_scissors/demo_30fps.h5"
USD = ASSETS + "scissors_flattened.usd"
MODEL_SHA = "45311f78d3c82d0d4b54050b68477cf26a66f67b924e25156bb9bc93848c20db"
CONSTANTS_SHA = "4281c40f43f2b17af036e5b9b73f84542ab4186df5914f7380367ca37401e855"
MODEL_PIN = "829c9aa27cb12c3a49d5b8d63c9869dd8f656702"
ORIGINAL_PINS = {
    "retargeted_120fps.h5": "e0d0523b7701447210de11d915f3af774566d2bc543ee2dc3ccd0af3a3e353d7",
    "metrics.json": "8651dd4aa2ca7c7a0994fbc1608b52fcad9840e3fe90362ebf1c8ba941351c61",
    "run.json": "84d337ae50be5568b78cbdbe866bd76029d14215ef1de7e26285f25b9e9bf83d",
}
RESTORED_PINS = {
    "retargeted_120fps.h5": "b7951e92245189833be5f017a8abc4ccbbc139c768fa5edaa26bee2057cc73f5",
    "metrics.json": "73fa27b94f94e8404de37ed6e6d4fbcb16b415a888a1ece16965fd8d42f85cfb",
    "run.json": "d574b6141c6187b5be5f5171b1f9bca29e5b55e5c81a5f9ab453d71eb8e777e7",
}
# Retarget 52+50 handles are provenance only. Runtime uses the separate 25+25 sets.
INPUT_PINS = {
    NATIVE_DEMO: "0e8f3676762b04aae78af0cc0b323fe8907f9c957353fecb2fa08946faaf0757",
    DEMO: "7070e99cc2c828cdd483b6df6262ad593f2ff2e21be4fb44702421d21ed87873",
    REPO + "data/retargeted_traj/wujihand/scissors/retargeted_120fps.h5": "ed34df7d989be6cd26a1e4279810ab510bf5d3bb367080f7a5d11077bf61d70e",
    ASSETS + "scissors.urdf": "2d99ae94e2cbb3383e8fcbb4fd687f7bbe633a12528178460dc15ca189d43fd6",
    REPO + "data/keypoints/scissors_bottom_100_filtered_handles.npy": "8978f8690d1131855ae1f96311878c8604d3c13667db4deb9b23fec192eaac5b",
    REPO + "data/keypoints/scissors_top_100_filtered_handles.npy": "abf9e91aa938455134710c0ee517e1916c6d58dc3b317d8562aaca5b1be831f8",
    USD: "50facbab88961e0adfd0f5a7ab63a5d735be6c3677d5ff73e8ce793fddf27f20",
}
KP_PINS = {
    REPO + "data/keypoints/scissors_bottom_25.npy": "cdc2676449af662110f4c16355351729bae0c1affaa4b8039202807877b1b2e9",
    REPO + "data/keypoints/scissors_top_25.npy": "10544a71f284f4fd61552ffa963b42bd29039134b166377a22ad60cd88eaee1e",
}
CODE_PINS = {
    "scripts/arctic_hand2/retarget_scissors.py": "d1662f01125a352c3a406040e5cd72b4a9295ff25a40f8bc280c2be0cf2f2a52",
    "scripts/arctic_hand2/restore_scissors_feasibility.py": "4a0728ba1c49d2b16e8683b6fe7127bd2222e484dea629f55d3dfe731bfcd3d7",
    "scripts/arctic_hand2/test_scissors_feasibility.py": "21c43798f379dfab34132e265e08f7ecc40894637a8bbf448e6a5e8fb94a2a86",
    REPO + "source/regrind/regrind/robots/hand2_right_contract.py": MODEL_SHA,
    REPO + "source/regrind/regrind/retargeting/wujihand2_right_diagnostic_constants.py": CONSTANTS_SHA,
}
RIGHT_FACE_SHA = "88407e45785b12311b7d6837f0a87bb6b47667ee1cdd3bd69245dcf25068631b"
TIP_VERTICES = [744, 320, 443, 554, 671]
CHANGED = [228, 260, 261, 514, 520, 549, 554, 592, 594, 595, 598, 601, 604, 605, 607, 608, 609, 610]
MUTABLE = {"robot_pos", "robot_quat", "robot_joints", "robot_keypoints", "anchor_wrist_pos", "anchor_wrist_quat"}
SHAPES = {"robot_quat": (4,), "robot_pos": (3,), "robot_joints": (20,), "object_quat": (4,),
          "object_pos": (3,), "object_joint": (1,), "mano_joint_coords": (21, 3),
          "robot_keypoints": (21, 3), "anchor_wrist_pos": (3,), "anchor_wrist_quat": (4,), "time_seconds": ()}
METHOD = "scissors_hand27_minimum_unscaled_mixed_coordinate_L2_feasibility_projection"
CAPS = {"root_translation_m": .0005, "root_rotation_rad": .001, "joint_max_rad": .001,
        "wrist_translation_m": .0005, "wrist_rotation_rad": .001}
COUNTS = {"self": 310, "object": 810, "hand_table": 27, "object_table": 30}
BOUND = .002001
SMOOTH = ("joint_angular_acceleration_mean_abs_rad_s2", "joint_angular_jerk_rms_rad_s3")
TRACK = ("ordered_21_mean_m", "wrist_mean_m")
TABLE = {"height_m": .945, "size_xy_m": [1., 1.], "position_xy_m": [0., 0.], "thickness_m": .02,
         "welded": True, "added_positions": 0, "official_runtime_tabletop_m": .952, "runtime_equivalence_claimed": False}
COPIES = {"retargeted_120fps.h5": SOURCE + "/retargeted_120fps.h5", "demo_interp_120fps.h5": DEMO,
          "object/scissors_flattened.usd": USD, "proof/run.json": SOURCE + "/run.json",
          "proof/metrics.json": SOURCE + "/metrics.json",
          **{f"keypoints_{part}.npy": REPO + f"data/keypoints/scissors_{part}_25.npy" for part in ("bottom", "top")}}
FILES = set(COPIES) | {"manifest.json", "approval.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(path):
    path = Path(path)
    require(str(path).strip() and ".." not in path.parts, "Empty/traversing path")
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"Symlink path: {path}")
    return path


def local_path(base, relative):
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Nonlocal path")
    return safe_path(safe_path(base) / relative)


def regular_file(path):
    path = safe_path(path)
    st = path.stat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, f"Not an independent regular file: {path}")
    return path


def sha256(path):
    return hashlib.sha256(regular_file(path).read_bytes()).hexdigest()


def verify_hash(path, digest):
    require(sha256(path) == digest, f"SHA-256 mismatch: {path}")


def source_pins():
    return {**INPUT_PINS, **KP_PINS, **CODE_PINS,
            **{f"{ORIGINAL}/{k}": v for k, v in ORIGINAL_PINS.items()},
            **{f"{SOURCE}/{k}": v for k, v in RESTORED_PINS.items()}}


def load_model_contract():
    path = Path(__file__).with_name("hand2_right_contract.py")
    verify_hash(path, MODEL_SHA)
    verify_hash(path.parents[1] / "retargeting/wujihand2_right_diagnostic_constants.py", CONSTANTS_SHA)
    spec = importlib.util.spec_from_file_location("scissors_pinned_hand2_model", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    verify_hash(model.C.ASSET_DIR / "manifest.json", model.C.MANIFEST_SHA256)
    for rel, digest in json.loads((model.C.ASSET_DIR / "manifest.json").read_text())["generated_files_sha256"].items():
        verify_hash(local_path(model.C.ASSET_DIR, rel), digest)
    model.validate_asset()
    return model


def physics_assumptions():
    return {"root_controller_profile": "com_wrench_bounded_v4", "kp_pos": 300., "kd_pos": 30.,
            "kp_rot": 3., "kd_rot": .02, "pd_component_force_cap_N": 50., "pd_component_torque_cap_Nm": 1.,
            "caps_are_total_wrench_limits": False, "support": "root_only_whole_hand_weight_and_COM_moment_of_applied_PD_force",
            "finger_joint_gravity_compensation": False, "object_compensation": False,
            "hand_gravity": True, "object_gravity": True, "gravity_m_s2": [0., 0., -9.81],
            "actions": 26, "finger_residual_scale": .32, "finger_parameters": "uncalibrated_vendor_parameters",
            "object_joint_axis": [0., 0., -1.], "object_joint_friction": .2, "runtime_joint_limits_rad": [0., 2.],
            "runtime_collision_count": "unknown_until_actual_USD_enumeration", "drake_collision_hulls": 30,
            "tabletop_m": .952, "table_center_xy_m": [0., 0.], "drake_tabletop_m": .945,
            "fixed_start": True, "randomization": False, "augmentation": False, "normal_six_terminations": True,
            "table_contact_threshold_N": 10., "physics_certified": False, "cooked_hull_equivalence_certified": False}


def approval_fields():
    """Parent adds approved, synthetic_test, authorization entries and actual quotes.

    This is a schema, not authentication. The parent must bind the real conversation.
    """
    return {"schema": "hand2_scissors_reference_approval_v1", "case": "scissors", "issuer_role": "parent",
            "scope": "Scissors runtime reference and authorized fresh formal training",
            "formal_training": {"num_envs": 4096, "iterations": 20000, "seed": 42, "resume": False},
            "task_ids": ["Regrind-WujiHand2-Scissors-Official-S-v0", "Regrind-WujiHand2-Scissors-Official-S-Play-v0"],
            "experiment_name": "wujihand2_scissors_official_s",
            "namespace": NAMESPACE, "model_pin": MODEL_PIN, "model": load_model_contract().provenance(),
            "source_pins": source_pins(), "window_30fps_closed": WINDOW, "native_frames_120fps": FRAMES,
            "runtime_controls": CONTROLS, "runtime_seconds": 9., "speed_factor": SPEED,
            "physics_assumptions": physics_assumptions()}


def validate_approval(receipt, *, allow_synthetic=False):
    require(isinstance(receipt, dict) and receipt.get("approved") is True, "Explicit parent approval required")
    for key, value in approval_fields().items():
        require(receipt.get(key) == value, f"Approval mismatch: {key}")
    auth = receipt.get("authorization", {})
    for role, ref in (("remote_request", "d78443d1"), ("confirmation", "8ca4b1dd"), ("confirmation_context", "c96b8f4a")):
        item = auth.get(role, {})
        require(item.get("message_id") == ref, f"Wrong authorization reference: {role}")
        for key in ("quote", "context_ref"):
            require(isinstance(item.get(key), str) and item[key].strip(), f"Missing actual {role}/{key}")
        if receipt.get("synthetic_test") is True:
            require(allow_synthetic and "synthetic TEST" in item["quote"], "Synthetic receipt forbidden")
        else:
            require(receipt.get("synthetic_test") is False and not re.search(
                r"\b(synthetic|mock|placeholder|todo)\b", item["quote"] + item["context_ref"], re.I), "Not actual authorization")
    return receipt


def test_owner(path, owner):
    require(isinstance(owner, tempfile.TemporaryDirectory), "Synthetic test requires owning TemporaryDirectory")
    root = safe_path(owner.name)
    require(root.parent == Path("/tmp") and root.name.startswith("hand2-scissors-") and root.is_dir()
            and safe_path(path).is_relative_to(root) and safe_path(path) != root, "Not owned Scissors temporary fixture")


def inspect_h5(file):
    """Reject external/soft/aliased/virtual storage before reading any payload."""
    addresses = set()
    def walk(group):
        for name in group:
            require(isinstance(group.get(name, getlink=True), h5py.HardLink), "H5 links forbidden")
            item = group[name]
            info = h5py.h5o.get_info(item.id)
            require(info.addr not in addresses and info.rc == 1, "H5 aliases forbidden")
            addresses.add(info.addr)
            if isinstance(item, h5py.Group):
                walk(item)
            else:
                require(not item.is_virtual and item.external is None, "External/virtual H5 storage")
                value = item[()]
                if item.dtype.kind in "fci":
                    require(np.isfinite(value).all(), "Nonfinite H5")
    walk(file)


def validate_demo(file, frames=FRAMES):
    inspect_h5(file)
    require(frames in (FRAMES, NATIVE_FRAMES) and set(file) == {"qpos", "mano_joint_coord", "mano_verts", "mano_faces", "meta"}, "Official demo schema")
    meta = {k: v[()] for k, v in file["meta"].items()}
    def text(k):
        v = meta[k]
        return v.decode() if isinstance(v, bytes) else str(v)
    for key, value in {"object_name": "scissors", "task_name": "arctic_leap_scissors", "manipulation_type": "use",
                       "object_scale": "1", "object_scale_suffix": ""}.items():
        require(text(key) == value, f"Official metadata: {key}")
    for key, value in {"start_timestep": 25, "end_timestep": 185, "traj_idx": 2, "traj_len": 661, "object_scale_float": 1.}.items():
        require(meta[key] == value, f"Official metadata: {key}")
    for key, suffix in (("mano_traj_path", ".mano.npy"), ("object_traj_path", ".object.npy"), ("processed_traj_path", ".npy")):
        require(text(key).endswith("/s02/scissors_use_01" + suffix), "Wrong native sequence")
    arrays = {k: file[k][()] for k in ("qpos", "mano_joint_coord", "mano_verts", "mano_faces")}
    for key, shape, dtype in (("qpos", (frames, 30), "float64"), ("mano_joint_coord", (frames, 21, 3), "float64" if frames == FRAMES else "float32"),
                               ("mano_verts", (161, 778, 3), "float32"), ("mano_faces", (1538, 3), "uint32")):
        require(arrays[key].shape == shape and arrays[key].dtype == np.dtype(dtype), f"Official shape/dtype: {key}")
    require(hashlib.sha256(arrays["mano_faces"].tobytes()).hexdigest() == RIGHT_FACE_SHA, "Not pinned MANO RIGHT faces")
    tips = arrays["mano_verts"][:, TIP_VERTICES]
    if frames == FRAMES:
        tips = np.stack([np.interp(np.arange(FRAMES)/4, np.arange(161), tips[:, j, d])
                         for j in range(5) for d in range(3)], axis=1).reshape(FRAMES, 5, 3)
    require(np.array_equal(arrays["mano_joint_coord"][:, 16:21], tips), "Native RIGHT mesh tips changed")
    require(np.allclose(np.linalg.norm(arrays["qpos"][:, 3:7], axis=1), 1, atol=1e-6, rtol=0), "Demo quaternion")
    return arrays


def names(model):
    return {"robot_joint_names": model.JOINT_NAMES, "mano_joint_names": model.C.MANO_JOINTS,
            "mano_keypoint_names": list(model.C.MANO_TO_ROBOT_MAPPING),
            "robot_keypoint_link_names": list(model.C.MANO_TO_ROBOT_MAPPING.values())}


def robot_metadata(model):
    return {"case": "scissors", "hand": "right", "sequence": "s02/scissors_use_01", "fps": 120,
            "source_fps": 30, "source_frames": 161, "duration_seconds": 640/120,
            "root_frame": "r_mount", "anchor_wrist_frame": "r_wrist", "anchor_wrist_source": "robot_FK",
            "model_pin": MODEL_PIN, "robot_name": "wujihand2_right_diagnostic_parent_weld", "collision_policy": "parent-weld",
            "quaternion_order": "wxyz", "quality_pass": False, "approved_for_training": False, "physics_certified": False,
            "q_layout": "root_wxyz_xyz[0:7],hand20[7:27],object_wxyz_xyz_joint[27:35]",
            "range_120fps_closed": [0, 640], "source_window_30fps_closed": WINDOW,
            "source_time_seconds_closed": [25/30, 185/30], "source_upstream_commit": "38347a9e30184620df04e19c63c7c72378cae103",
            "window_role": "full_native_diagnostic_NOT_training", "mano_joint_coords_order": "numeric MANO 0..20",
            "robot_keypoints_order": "mano_keypoint_names insertion order", "mapping": json.dumps(model.C.MANO_TO_ROBOT_MAPPING),
            "source_sha256": json.dumps(INPUT_PINS, sort_keys=True), "table_geometry": json.dumps(TABLE),
            "restoration_stage": METHOD, "restoration_source_h5_sha256": ORIGINAL_PINS["retargeted_120fps.h5"],
            "restoration_quality_pass": True}


def validate_robot(file, model, demo, run):
    inspect_h5(file)
    require(set(file) == set(SHAPES) | set(names(model)), "Robot dataset inventory")
    attrs = robot_metadata(model)
    require(set(file.attrs) == set(attrs), "Robot metadata inventory")
    for key, value in attrs.items():
        require(np.array_equal(file.attrs[key], value), f"Historical metadata/false flag: {key}")
    for key, tail in SHAPES.items():
        require(file[key].shape == (FRAMES, *tail) and file[key].dtype == np.dtype("float64"), f"Robot shape/dtype: {key}")
    for key, value in names(model).items():
        require(file[key].asstr()[:].tolist() == list(value), f"Wrong official named order: {key}")
    for key in ("robot_quat", "object_quat", "anchor_wrist_quat"):
        require(np.allclose(np.linalg.norm(file[key][()], axis=1), 1, atol=1e-6, rtol=0), "Robot quaternion")
    require(np.array_equal(file["time_seconds"][()], np.arange(FRAMES)/FPS), "Native timing changed")
    for key, expected in (("mano_joint_coords", demo["mano_joint_coord"]), ("object_pos", demo["qpos"][:, :3]),
                          ("object_quat", demo["qpos"][:, 3:7]), ("object_joint", demo["qpos"][:, 7:8])):
        require(np.array_equal(file[key][()], expected), f"Native human/object8 changed: {key}")
    opening = file["object_joint"][()]
    require(float(opening.max()) == .7717662453651428 and float(opening.min()) == .08384069800376892 and np.count_nonzero(opening > .5) == 243,
            "Original opening must not be clamped to legacy .5 rad")
    q = file["robot_joints"][()]
    for i, p in enumerate(model.PARAMETERS.values()):
        require(np.all(q[:, i] >= p["lower"]-1e-6) and np.all(q[:, i] <= p["upper"]+1e-6), "Hand joint limits")
    keep = np.ones(FRAMES, dtype=bool)
    keep[CHANGED] = False
    hashes = {k: hashlib.sha256((file[k][()][keep] if k in MUTABLE else file[k][()]).tobytes()).hexdigest() for k in file}
    require(hashes == run["preservation"]["immutable_dataset_rows_sha256"], "Immutable 623/object8/human/timing proof changed")


def evaluation(file, model):
    q, human = file["robot_joints"][()], file["mano_joint_coords"][()]
    return {SMOOTH[0]: float(np.abs(np.diff(q, n=2, axis=0)*FPS**2).mean()),
            SMOOTH[1]: float(np.sqrt(np.mean((np.diff(q, n=3, axis=0)*FPS**3)**2))),
            TRACK[0]: float(np.linalg.norm(file["robot_keypoints"][()] - human[:, [int(k) for k in model.C.MANO_TO_ROBOT_MAPPING]], axis=2).mean()),
            TRACK[1]: float(np.linalg.norm(file["anchor_wrist_pos"][()] - human[:, 0], axis=1).mean())}


def validate_evaluation(values, metrics, label):
    record = metrics["evaluation"]["original_full641" if label == "baseline" else "candidate_full641"]
    require(record["slice_120fps_half_open"] == [0, FRAMES] and record["frames"] == FRAMES and record["fps"] == FPS, "Full evaluation scope")
    for key, value in values.items():
        expected = record[key] if key in SMOOTH else record["tracking"][key]
        require(np.isclose(value, expected, rtol=1e-10, atol=1e-12)
                and np.isclose(value, metrics["regression_checks"][key][label], rtol=1e-10, atol=1e-12), "Full evaluation mismatch")


def validate_quality(run, metrics):
    for r in (run, metrics):
        require(r["case"] == "scissors" and r["status"] == "quality_pass" and r["restoration_stage"] == METHOD
                and all(r[k] is False for k in ("quality_pass", "approved_for_training", "physics_certified", "source_quality_pass"))
                and r["restoration_quality_pass"] is True and r["candidate_nonlinear_quality_pass"] is True
                and r["frames_120fps"] == FRAMES and r["trajectory_fps"] == FPS and r["table_geometry"] == TABLE, "Stage A flags/scope")
    require(run["source_sha256"] == ORIGINAL_PINS and run["source_directory"] == ORIGINAL
            and run["frames_completed"] == FRAMES and run["range_120fps_closed"] == [0, 640]
            and run["changed_frames"] == run["bad_frames"] == CHANGED and run["projection_qp_count"] == 18
            and run["restoration_invocations"] == 1 and run["retarget_invocations"] == run["render_invocations"] == 0
            and run["worker_returncode"] == 0 and run["timed_out"] is False, "Stage A invocation/counters")
    require(run["q_layout"] == "hand[0:27],locked_object8[27:35]" and run["object_top_fk"] == "bottom @ Rz(-object_joint); zero offset", "Stage A articulated layout")
    for key in ("retargeted_120fps.h5", "metrics.json"):
        require(run["output_sha256"][key] == RESTORED_PINS[key], "Stage A output binding")
    for key in ("input_model_sha256", "code_sha256"):
        require(run[key] == run[key+"_after"], "Stage A input/code changed")
    require(run["protected_sha256_before"] == run["protected_sha256_after"] == {f"{ORIGINAL}/{k}": v for k, v in ORIGINAL_PINS.items()}
            and run["inputs_model_code_unchanged"] is True and run["full_h5_preservation_verified"] is True, "Stage A preservation")
    for key, digest in INPUT_PINS.items():
        require(run["input_model_sha256"][key] == digest, "Stage A input pin")
    for key, digest in CODE_PINS.items():
        if key.startswith("scripts/arctic_hand2/"):
            require(run["code_sha256"][key] == digest, "Stage A code pin")
    require(run["parameters"] == {"solver": "StrictMosekSolver", "max_qps_per_row": 3, "max_qps_total": 54,
            "cumulative_original_row_delta_caps": CAPS, "nonlinear_bound_m": BOUND, "linear_tolerance_m": .002,
            "near_query_m": .1, "trust_norm": .2, "timeout_seconds": 1800, "retry_fallback_clipping": False,
            "solver_options_changed": False}, "Projection contract")
    rows = run["projection_frames"]
    require([r["frame"] for r in rows] == CHANGED, "Projection rows")
    for r in rows:
        require(r["status"] == "restored" and r["qp_attempts"] == len(r["iterations"]) == 1, "Projection attempts")
        it = r["iterations"][0]
        require(it["status"] == "feasible" and it["delta_measured"] is True, "Projection status")
        near = it["near_query"]
        require(near["full_verified_support_pairs"] == 30 and 0 <= near["skipped_verified_support_pairs"] <= 30, "Support subset")
        for key, cap in CAPS.items():
            require(np.isfinite(it["cumulative_delta_from_original"][key]) and 0 <= it["cumulative_delta_from_original"][key] <= cap, "Original-relative cap")
    for key, cap in CAPS.items():
        require(run["cumulative_delta_maxima"][key] == max(r["iterations"][0]["cumulative_delta_from_original"][key] for r in rows)
                and run["cumulative_delta_maxima"][key] <= cap, "Delta maxima")
    require(metrics["penetration_bound_m"] == BOUND and metrics["object_table_policy"] == "measured_report_only_inherited_support_exemption", "Nonlinear bound/support")
    validity = metrics["validity"]
    require(all(validity[k] is True for k in ("finite", "normalized_quaternions", "hand_joint_limits", "object8_bit_exact_locked",
                "fk_consistent", "nonlinear_self", "nonlinear_object", "nonlinear_hand_table")), "Nonlinear validity")
    for key, cap in (("max_quaternion_norm_error", 1e-6), ("joint_lower_violation_max_rad", 1e-6),
                     ("joint_upper_violation_max_rad", 1e-6), ("object_max_abs_error", 0.), ("fk_max_abs_error", 1e-10)):
        require(np.isfinite(validity[key]) and 0 <= validity[key] <= cap, "Nonlinear numeric validity")
    for kind, count in COUNTS.items():
        item = metrics["collision"][kind]
        a, d = (np.asarray(item[k]) for k in ("per_frame_max_penetration_m", "per_frame_min_signed_distance_m"))
        require(a.shape == d.shape == (FRAMES,) and np.isfinite(a).all() and np.isfinite(d).all()
                and np.array_equal(a, np.maximum(-d, 0)) and item["max_penetration_m"] == float(a.max())
                and item["queried_pair_count"] == count*FRAMES and run["collision_filter_contract"][kind+"_candidates"] == count
                and item["frames_over_bound"] == int(np.count_nonzero(a > BOUND)), "All-pair proof/counters")
        if kind != "object_table":
            require(np.all(a <= BOUND) and item["count_over_bound"] == 0, "Nonlinear penetration bound")
    require(run["support_exemption"] == {"inherited_only": True, "object_table_exempt_count": 30, "hand_exempt_count": 0,
            "all_hand_pairs_checked": 1188, "additional_filters": False, "object_table_policy": "report_only"}, "Support exemption")
    require(run["preservation"] == metrics["preservation"] and run["preservation"]["unchanged_robot_rows"] == 623
            and run["preservation"]["all_original_attributes_and_storage_exact"] is True
            and run["preservation"]["independent_inode"] is True, "Immutable rows proof")
    require(run["evaluation_contract"] == {"scope": "this bounded correction only; not universal user policy",
            "full_range_120fps_half_open": [0, 641], "fps": 120, "smoothness_ratio_bound": 1.01,
            "zero_smoothness_baseline_absolute_bound": 1e-12, "mean_tracking_increase_bound_m": .0001}, "Engineering evaluation bounds")
    require(run["regression_checks"] == metrics["regression_checks"] and set(metrics["regression_checks"]) == set(SMOOTH+TRACK), "Regression proof")
    for key, r in metrics["regression_checks"].items():
        upper = (r["baseline"]*1.01 if r["baseline"] else 1e-12) if key in SMOOTH else r["baseline"]+.0001
        require(all(np.isfinite(r[k]) for k in ("baseline", "candidate", "upper_bound", "difference"))
                and r["pass"] is True and r["candidate"] <= upper and r["upper_bound"] == upper
                and r["difference"] == r["candidate"]-r["baseline"], "Engineering preservation failed")
    require(run["missing_visual_dependencies"] == [ASSETS+"material_0.mtl"], "Original missing visual material provenance")
    dif = run["physical_differences"]
    require(dif == metrics["physical_differences"] and dif["object_joint_max_rad"] == .7717662453651428
            and dif["object_joint_outside_drake_urdf_frames"] == 243 and dif["drake_joint_limits_rad"] == [0., .5]
            and dif["official_runtime_both_COM_x_m"] == -.01 and dif["source_sha256"] == INPUT_PINS[USD]
            and all(dif[k] is False for k in ("runtime_equivalence_claimed", "physics_certified", "cooked_hull_equivalence_certified")), "Physical differences must remain explicit")


def expected_manifest():
    return {"schema": "hand2_scissors_runtime_v1", "namespace": NAMESPACE, "source_pins": source_pins(),
            "frames_120fps": FRAMES, "native_window_30fps_closed": WINDOW, "runtime_controls": CONTROLS,
            "runtime_seconds": 9., "speed_factor": SPEED, "native_h5_stretched": False,
            "runtime_keypoint_counts": {"bottom": 25, "top": 25}, "retarget_keypoint_counts": {"bottom": 52, "top": 50},
            "right_faces_sha256": RIGHT_FACE_SHA, "tip_vertices": TIP_VERTICES, "physics_assumptions": physics_assumptions(),
            "historical_quality_pass": False, "restoration_quality_pass": True, "physics_certified": False,
            "missing_visual_dependencies": [ASSETS+"material_0.mtl"], "visual_fidelity_validated": False}


def bundle_path(data_root=None):
    return local_path(data_root if data_root is not None else os.environ["REGRIND_DATA_DIR"], NAMESPACE)


def validate_bundle(directory, *, test_fixture_owner=None):
    directory = safe_path(directory)
    if test_fixture_owner is not None:
        test_owner(directory, test_fixture_owner)
    require(directory.is_dir(), "Missing Scissors bundle")
    inventory = set()
    for path in directory.rglob("*"):
        safe_path(path)
        if path.is_dir():
            require(path.relative_to(directory).as_posix() in {"object", "proof"}, "Extra bundle directory")
        else:
            regular_file(path)
            inventory.add(path.relative_to(directory).as_posix())
    require(inventory == FILES, "Bundle must contain exactly nine files")
    manifest = json.loads((directory / "manifest.json").read_text())
    require(set(manifest) == set(expected_manifest()) | {"model", "joint_names", "approval_sha256", "output_sha256"}, "Manifest field inventory")
    for key, value in expected_manifest().items():
        require(manifest.get(key) == value, f"Manifest mismatch: {key}")
    hashes = manifest["output_sha256"]
    require(set(hashes) == FILES - {"manifest.json"}, "Manifest hash inventory")
    for rel, digest in hashes.items():
        verify_hash(local_path(directory, rel), digest)
    for rel, source in COPIES.items():
        require(hashes[rel] == source_pins()[source], f"Not byte-identical pinned copy: {rel}")
    receipt = json.loads((directory / "approval.json").read_text())
    validate_approval(receipt, allow_synthetic=test_fixture_owner is not None)
    require(manifest["approval_sha256"] == hashes["approval.json"], "Approval hash mismatch")
    model = load_model_contract()
    require(manifest["model"] == model.provenance() and manifest["joint_names"] == list(model.JOINT_NAMES), "Runtime model/order")
    run, metrics = (json.loads((directory / "proof" / n).read_text()) for n in ("run.json", "metrics.json"))
    validate_quality(run, metrics)
    for part in ("bottom", "top"):
        points = np.load(directory / f"keypoints_{part}.npy", allow_pickle=False)
        require(points.shape == (25, 3) and np.isfinite(points).all(), "Runtime requires 25+25 keypoints")
    with h5py.File(directory / "demo_interp_120fps.h5", "r") as f:
        demo = validate_demo(f)
    with h5py.File(directory / "retargeted_120fps.h5", "r") as f:
        validate_robot(f, model, demo, run)
        validate_evaluation(evaluation(f, model), metrics, "candidate")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args(argv)
    manifest = validate_bundle(args.bundle or bundle_path())
    print(json.dumps({"portable_check_passed": True, "frames": manifest["frames_120fps"], "physics_certified": False}))


if __name__ == "__main__":
    main()
