"""Portable Spatula reference validation, usable BEFORE AppLauncher.

No Drake, original workspace data, other case bundle or approval is needed here.
Pinned restoration evidence is checked, not recomputed collision geometry. Source
quality_pass/approved_for_training remain false; no table/physics/visual certificate.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import h5py
import numpy as np

HELPER_SHA = "618e2867894954a68b2f923a08cbc2605dca038abab9e6784f8bbf1b8d248bfc"
_helper_path = Path(__file__).with_name("hand2_case_contract.py")
if any(p.is_symlink() for p in (_helper_path, *_helper_path.parents)) or hashlib.sha256(_helper_path.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError("Pinned pure case helper changed")
_spec = importlib.util.spec_from_file_location("spatula_pure_case_helpers", _helper_path)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
# Only case-independent pure helpers; never change H.CASES or call its validators.
require, safe_path, local_path = H.require, H.safe_path, H.local_path
sha256, verify_hash, validate_arrays = H.sha256, H.verify_hash, H.validate_arrays
ROBOT_SHAPES, HUMAN_SHAPES, CONSTANT_DATASETS = H.ROBOT_SHAPES, H.HUMAN_SHAPES, H.CONSTANT_DATASETS
MODEL_PIN, FROZEN_REL, FROZEN_SHA, RHO_REL, RHO_SHA = H.MODEL_PIN, H.FROZEN_REL, H.FROZEN_SHA, H.RHO_REL, H.RHO_SHA
MODEL_SHA = "45311f78d3c82d0d4b54050b68477cf26a66f67b924e25156bb9bc93848c20db"
CONSTANTS_SHA = "4281c40f43f2b17af036e5b9b73f84542ab4186df5914f7380367ca37401e855"
CASE_ID = "shovel_spatula_R_s1t1_sw05"
TAKE = "TactiDex_Shovel_S1H01A0036T1_R"
FRAMES, CONTROLS, BOUND = 2229, 558, 0.001001
WINDOW30, WINDOW120, ANNOTATION30 = [0, 557], [0, 2229], [197, 238]
CHANGED = [470, 471, 472]
SOURCE = "data/tactidex/retargeted/wujihand2_diagnostic/spatula_parent_weld_meshtip_feasible_v1"
ORIGINAL = "data/tactidex/retargeted/wujihand2_diagnostic/spatula_parent_weld_meshtip_v1/retargeted_120fps.h5"
DEMO = f"data/tactidex/regrind_demo_meshtip_hand2_right_v1/{TAKE}/demo_30fps.h5"
KEYPOINTS = "data/tactidex/keypoints/spatula_100_filtered_shovel.npy"
OBJECT_DIR = "data/tactidex/assets/spatula"
NAMESPACE = "hand2_right/spatula_meshtip_v1"
H5_SHA = "daf4a1489389ef31d4155f6aea10ec397514267ba434b3e028575b7a9cff9c71"
METRICS_SHA = "ce52d40f083a63187e1ef80a82b1c3f5461eee6134f2528bf7ec55477a9a03a3"
RUN_SHA = "4562e308a2ebc74c72c2f1185d59206daa5e8ddc16098baa0f8533a5ac854b34"
ORIGINAL_SHA = "b9c4f27c7058e6e2a4ef2f491c499915727cd1fb1d81ae1a283d79c96e6cfd18"
DEMO_SHA = "546fd9065ea9ac16ef47ec68b8b693cf014b73f75f5f55b53f3b016caaac8d23"
KEYPOINT_SHA = "538e3b5223d2796b0ebb7ef6a85b5c7de349afd77edce4c7619a2e57260ec9fe"
URDF_SHA = "37cc7831881353b40276735040dfe16551ecc9841bc546ab1bb755c881b934c6"
OBJECT_PAYLOAD_SHA = "9f26c1ec50a51e16f45d8ed8219f456a8a13ffae2734d82d7270cdb45bb16e16"
MISSING_VISUAL = ["3DModel_Custom.mtl"]
RESTORATION_STAGE = "minimum_mixed_coordinate_L2_feasibility_projection"
SMOOTH = ("joint_angular_acceleration_mean_abs_rad_s2", "joint_angular_jerk_rms_rad_s3")
TRACK = ("ordered_21_mean_m", "wrist_mean_m")
DELTA_CAPS = {"root_translation_m": 0.0005, "root_rotation_rad": 0.001, "joint_max_rad": 0.001}
TABLE_RECORD = {"contract_table_height_m": 0.0, "drake_table_height": None,
                "table_geometry_present": False, "table_clearance_certified": False,
                "baseline_no_table_geometry_unchanged": True}
# SHA256 of contiguous float64 arrays from the pinned demo + pinned RHO factor4.
# These bind even human wrist orientation without shipping original workspace data.
HUMAN_ARRAY_SHA256 = {
    "mano_joint_coords": "ad704db5b6a2f45a967896746cb26dfebaacb9efdb1cb7f46407f9a173ba8ee2",
    "wrist_pos": "e85937a8915be6ed9ca9d77bf7bb541f6d94bc1d098088f3822574943bfc9d9d",
    "wrist_quat": "91653b245954a266cc64b8f4adfa524088b7bdf8eb3ff1fea337ecd504dcf8b1",
    "obj_pos": "ca28a13cf6235bbf4f02168b552f966abb50b683aacef53be3b090bc48fe5ae0",
    "obj_quat": "115c69ae1bd81861f8cbbc9ecb08eeefce198c7662aa09236c9f3ed1b2cd26aa",
}


def source_pins():
    return {FROZEN_REL: FROZEN_SHA, RHO_REL: RHO_SHA, DEMO: DEMO_SHA, ORIGINAL: ORIGINAL_SHA,
            KEYPOINTS: KEYPOINT_SHA, f"{OBJECT_DIR}/spatula.urdf": URDF_SHA,
            f"{SOURCE}/retargeted_120fps.h5": H5_SHA, f"{SOURCE}/run.json": RUN_SHA,
            f"{SOURCE}/metrics.json": METRICS_SHA,
            "repos/regrind/source/regrind/regrind/robots/hand2_case_contract.py": HELPER_SHA,
            "repos/regrind/source/regrind/regrind/robots/hand2_right_contract.py": MODEL_SHA,
            "repos/regrind/source/regrind/regrind/retargeting/wujihand2_right_diagnostic_constants.py": CONSTANTS_SHA}


def load_model_contract():
    verify_hash(Path(__file__).with_name("hand2_right_contract.py"), MODEL_SHA)
    verify_hash(Path(__file__).parents[1] / "retargeting/wujihand2_right_diagnostic_constants.py", CONSTANTS_SHA)
    model = H.load_model_contract()
    manifest_path = local_path(model.C.ASSET_DIR, "manifest.json")
    verify_hash(manifest_path, model.C.MANIFEST_SHA256)
    asset = json.loads(manifest_path.read_text())
    for relative, digest in asset["generated_files_sha256"].items():
        verify_hash(local_path(model.C.ASSET_DIR, relative), digest)
    model.validate_asset()
    return model


def physics_assumptions():
    return {"root_controller_profile": "com_wrench_bounded_v4", "kp_pos": 300.0, "kd_pos": 30.0,
            "kp_rot": 3.0, "kd_rot": 0.02, "pd_component_force_cap_N": 50.0,
            "pd_component_torque_cap_Nm": 1.0, "caps_are_total_wrench_limits": False,
            "support": "root_only_ideal_whole_hand_weight_and_COM", "object_compensation": False,
            "hand_gravity": True, "object_gravity": True, "gravity_m_s2": [0.0, 0.0, -9.81],
            "finger_parameters": "uncalibrated_vendor_parameters", "actions": 26,
            "finger_residual_scale": 0.32, "object_mass_kg": 0.15, "object_collision_hulls": 3,
            "tabletop_m": 0.012, "table_center_x_m": -0.15, "speed_factor": 1.0,
            "fixed_start": True, "randomization": False, "augmentation": False,
            "normal_train_play_terminations": True, "physics_certified": False,
            "table_clearance_certified": False, "visual_fidelity_validated": False}


def approval_fields():
    """Parent receipt API: add approved=true, synthetic_test=false, user_quote and
    user_message_ref (the actual authorization and its external message reference).
    This module never creates approvals and cannot authenticate the external user.
    """
    return {"schema": "hand2_spatula_reference_approval_v1", "case": "spatula", "case_id": CASE_ID,
            "issuer_role": "parent", "model_pin": MODEL_PIN, "source_pins": source_pins(),
            "object_mesh_payload_sha256": OBJECT_PAYLOAD_SHA, "human_array_sha256": HUMAN_ARRAY_SHA256,
            "window_30fps_closed": WINDOW30, "window_120fps_half_open": WINDOW120,
            "namespace": NAMESPACE, "scope": "Spatula runtime reference and authorized formal training", "physics_assumptions": physics_assumptions()}


def validate_approval(receipt, *, allow_synthetic=False):
    require(isinstance(receipt, dict) and receipt.get("approved") is True, "Missing explicit parent approval receipt")
    for key, value in approval_fields().items():
        require(receipt.get(key) == value, f"Approval mismatch: {key}")
    for key in ("user_quote", "user_message_ref"):
        require(isinstance(receipt.get(key), str) and receipt[key].strip(), f"Missing actual authorization: {key}")
    if receipt.get("synthetic_test") is True:
        require(allow_synthetic and "synthetic TEST" in receipt["user_quote"], "Synthetic receipt forbidden")
    else:
        require(receipt.get("synthetic_test") is False, "Explicit synthetic_test=false required")
        require(not re.search(r"\b(synthetic|mock|placeholder|todo)\b", receipt["user_quote"] + " " + receipt["user_message_ref"], re.I),
                "Synthetic/placeholder authorization is not an actual user quote")
    return receipt


def validate_robot(file, model):
    validate_arrays(file, ROBOT_SHAPES, FRAMES)
    require(set(file) == set(ROBOT_SHAPES) | CONSTANT_DATASETS, "Unexpected robot datasets")
    attrs = {"case": "spatula", "root_frame": "r_mount", "anchor_wrist_frame": "r_wrist", "fps": 120,
             "robot_name": "wujihand2_right_diagnostic_parent_weld", "quaternion_order": "wxyz",
             "collision_policy": "parent-weld", "quality_pass": False, "approved_for_training": False,
             "restoration_quality_pass": True, "restoration_source_h5_sha256": ORIGINAL_SHA,
             "restoration_stage": RESTORATION_STAGE, "window_role": "full_diagnostic_NOT_training",
             "historical_compliance": "spatula_noncompliant_full", "source_frames_30fps": CONTROLS,
             "table_geometry_present": False, "table_clearance_certified": False,
             "mano_joint_coords_order": "numeric MANO 0..20", "robot_keypoints_order": "mano_keypoint_names insertion order"}
    for key, expected in attrs.items():
        require(file.attrs.get(key) == expected, f"Historical H5 metadata changed: {key}")
    require(np.array_equal(file.attrs.get("range_120fps_closed"), [0, FRAMES - 1])
            and np.array_equal(file.attrs.get("native_window_30fps_closed"), WINDOW30), "Historical range changed")
    require(json.loads(file.attrs["table_geometry"]) == TABLE_RECORD, "Table certification changed")
    require(json.loads(file.attrs["mapping"]) == model.C.MANO_TO_ROBOT_MAPPING, "Wrong mapping metadata")
    for key, expected in (("robot_joint_names", model.JOINT_NAMES), ("mano_joint_names", model.C.MANO_JOINTS),
                          ("mano_keypoint_names", model.C.MANO_TO_ROBOT_MAPPING),
                          ("robot_keypoint_link_names", model.C.MANO_TO_ROBOT_MAPPING.values())):
        require(file[key].asstr()[:].tolist() == list(expected), f"Wrong official order: {key}")
    q = file["robot_joints"][()]
    for i, params in enumerate(model.PARAMETERS.values()):
        require(np.all(q[:, i] >= params["lower"] - 1e-6) and np.all(q[:, i] <= params["upper"] + 1e-6), "Joint limits")


def validate_human_arrays(payload):
    require(set(payload) == set(HUMAN_SHAPES), "Unexpected human arrays")
    for key, tail in HUMAN_SHAPES.items():
        value = np.asarray(payload[key])
        require(value.shape == (FRAMES, *tail) and value.dtype == np.dtype("float64") and np.isfinite(value).all(), "Human schema/nonfinite")
        require(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == HUMAN_ARRAY_SHA256[key], f"Human interpolation changed: {key}")


def evaluation(file, model):
    joints, human = file["robot_joints"][()], file["mano_joint_coords"][()]
    ordered = human[:, [int(k) for k in model.C.MANO_TO_ROBOT_MAPPING]]
    return {SMOOTH[0]: float(np.abs(np.diff(joints, n=2, axis=0) * 120**2).mean()),
            SMOOTH[1]: float(np.sqrt(np.mean((np.diff(joints, n=3, axis=0) * 120**3)**2))),
            TRACK[0]: float(np.linalg.norm(file["robot_keypoints"][()] - ordered, axis=2).mean()),
            TRACK[1]: float(np.linalg.norm(file["anchor_wrist_pos"][()] - human[:, 0], axis=1).mean())}


def validate_evaluation(values, metrics, label):
    record = metrics["evaluation"][label]["full"]
    field = "baseline" if label == "baseline_recomputed" else "candidate"
    require(record["slice_120fps_half_open"] == WINDOW120 and record["frames"] == FRAMES, "Not full-scope evaluation")
    for key, value in values.items():
        expected = record[key] if key in SMOOTH else record["tracking"][key]
        require(np.isfinite(value) and np.isclose(value, expected, rtol=1e-10, atol=1e-12)
                and np.isclose(value, metrics["regression_checks"]["full"][key][field], rtol=1e-10, atol=1e-12), "Recomputed full-scope regression differs")


def validate_quality(run, metrics):
    """Explicit Spatula restoration proof: false historical quality is intentional."""
    for record in (run, metrics):
        require(record["case"] == "spatula" and record["status"] == "quality_pass"
                and record["restoration_quality_pass"] is True and record["source_quality_pass"] is False
                and record["quality_pass"] is False and record["restoration_stage"] == RESTORATION_STAGE, "Restoration flags changed")
        require(all(record[k] is False for k in ("approved", "approved_for_training", "C_approved")), "Historical approval changed")
        require(record["frames_120fps"] == FRAMES and record["range_120fps_closed"] == [0, FRAMES - 1], "Wrong full proof range")
        require(record["table_geometry"] == TABLE_RECORD, "No table geometry certification permitted")
    require(metrics["candidate_nonlinear_quality_pass"] is True and metrics["penetration_bound_m"] == BOUND, "Candidate quality/bound failed")
    validity = metrics["validity"]
    require(all(validity[k] is True for k in ("finite", "normalized_quaternions", "object_locked", "joint_limits",
                                            "nonlinear_self_collision", "nonlinear_object_collision")), "Nonlinear validity failed")
    for key, upper in (("max_quaternion_norm_error", 1e-6), ("object_max_abs_error", 0.0),
                       ("joint_lower_violation_max_rad", 1e-6), ("joint_upper_violation_max_rad", 1e-6)):
        require(np.isfinite(validity[key]) and 0 <= validity[key] <= upper, "Nonfinite/invalid nonlinear proof")
    require(validity["quaternion_tolerance"] == validity["joint_limit_tolerance_rad"] == 1e-6, "Changed validity tolerances")
    require(run["nonlinear_validity"] == validity and all(run[k] is True for k in
            ("full_h5_preservation_verified", "protected_outputs_unchanged", "inputs_model_code_unchanged")), "Preservation proof failed")
    require(run["changed_frames"] == run["bad_frames"] == CHANGED and run["projection_qp_count"] == 3
            and run["source_frames_30fps"] == CONTROLS and run["native_window_30fps_closed"] == WINDOW30, "Wrong restoration rows/QPs")
    require(run["retarget_invocations"] == run["render_invocations"] == 0 and run["restoration_invocations"] == 1, "Unexpected restoration execution")
    require(run["input_model_sha256"].get(ORIGINAL) == ORIGINAL_SHA
            and run["output_sha256"].get("retargeted_120fps.h5") == H5_SHA
            and run["output_sha256"].get("metrics.json") == METRICS_SHA, "Restoration source binding failed")
    require(run["parameters"]["cumulative_delta_caps_from_original"] == DELTA_CAPS, "Changed delta caps")
    require(run["evaluation_contract"] == {"scopes_120fps_half_open": {"full": WINDOW120}, "fps": 120,
            "smoothness_ratio_bound": 1.01, "zero_baseline_absolute_bound": 1e-12,
            "tracking_mean_increase_bound_m": 0.0001}, "Changed full-scope evaluation contract")
    require(run["support_exemption"] == {"all_hand_pairs_checked": 432, "support_exempt_count": 0}, "Support exemptions forbidden")
    require(run["missing_visual_dependencies"] == [f"{OBJECT_DIR}/3DModel_Custom.mtl"]
            and run["visual_fidelity_validated"] is False, "Visual proof mismatch")
    projections = run["projection_frames"]
    require([p["frame"] for p in projections] == CHANGED, "Wrong projection frames")
    for p in projections:
        require(p["status"] == "restored" and p["qp_attempts"] == p["iteration_count"] == 1 and len(p["iterations"]) == 1, "Wrong projection QP proof")
        for key, cap in DELTA_CAPS.items():
            value = p["iterations"][0]["cumulative_delta_from_original"][key]
            require(np.isfinite(value) and 0 <= value <= cap, "Original-relative projection cap failed")
    for kind, count in (("self", 310), ("object", 81)):
        require(run["collision_filter_contract"][f"{kind}_candidates"] == count, "Wrong collision candidate count")
        item = metrics["collision"][kind]
        values = np.asarray(item["per_frame_max_penetration_m"])
        distances = np.asarray(item["per_frame_min_signed_distance_m"])
        require(values.shape == distances.shape == (FRAMES,) and np.isfinite(values).all() and np.isfinite(distances).all()
                and np.array_equal(values, np.maximum(-distances, 0)) and np.all(values <= BOUND)
                and item["queried_pair_count"] == count * FRAMES and item["frames_over_bound"] == item["count_over_bound"] == 0
                and item["max_penetration_m"] == float(values.max()), "Full all-pair collision proof failed")
    require(run["regression_checks"] == metrics["regression_checks"] and set(metrics["regression_checks"]) == {"full"}, "Full regression proof mismatch")
    for key in (*SMOOTH, *TRACK):
        item = metrics["regression_checks"]["full"][key]
        b, v = item["baseline"], item["candidate"]
        upper = (b * 1.01 if b else 1e-12) if key in SMOOTH else b + 0.0001
        require(np.isfinite([b, v, item["difference"], item["upper_bound"]]).all() and min(b, v) >= 0
                and item["pass"] is True and item["upper_bound"] == upper and v <= upper
                and item["difference"] == v - b, "Full smoothness/tracking regression failed")
    for label in ("baseline_recomputed", "candidate"):
        require(set(metrics["evaluation"][label]) == {"full"}, "Unexpected evaluation scope")
        rec = metrics["evaluation"][label]["full"]
        validate_evaluation({key: rec[key] if key in SMOOTH else rec["tracking"][key] for key in (*SMOOTH, *TRACK)}, metrics, label)


def object_payload(directory):
    urdf = local_path(directory, "spatula.urdf")
    verify_hash(urdf, URDF_SHA)
    root = ET.parse(urdf).getroot()
    links = root.findall("link")
    require(root.get("name") == "spatula" and len(links) == 1 and links[0].get("name") == "bottom"
            and not root.findall("joint") and float(links[0].find("inertial/mass").get("value")) == 0.15, "Wrong rigid Spatula mass/body")
    collisions = links[0].findall("collision/geometry/mesh")
    meshes = {m.get("filename") for m in root.findall(".//mesh")}
    require(len(collisions) == 3 and {m.get("filename") for m in collisions} == {f"collision_parts/part_{i:03d}.obj" for i in range(3)}
            and meshes == {"spatula.obj", *(f"collision_parts/part_{i:03d}.obj" for i in range(3))}, "Expected three hulls and visual OBJ")
    hashes, missing = {}, set()
    for relative in sorted(meshes):
        mesh = local_path(directory, relative)
        hashes[relative] = sha256(mesh)
        for line in mesh.read_text().splitlines():
            if line.startswith("mtllib "):
                material_rel = line.split(maxsplit=1)[1]
                material = local_path(directory, material_rel)
                if not material.exists():
                    missing.add(material_rel)
                    continue
                hashes[material_rel] = sha256(material)
                for entry in material.read_text().splitlines():
                    if entry.startswith("map_"):
                        rel = entry.split(maxsplit=1)[1]
                        texture = local_path(directory, rel)
                        if texture.exists():
                            hashes[rel] = sha256(texture)
                        else:
                            missing.add(rel)
    require(sorted(missing) == MISSING_VISUAL, "Known missing material must remain missing; no repairs")
    full_hashes = {f"{OBJECT_DIR}/{k}": v for k, v in hashes.items()}
    require(hashlib.sha256(json.dumps(full_hashes, sort_keys=True).encode()).hexdigest() == OBJECT_PAYLOAD_SHA, "Object mesh payload pin mismatch")
    hashes["spatula.urdf"] = URDF_SHA
    return {"urdf": "object/spatula.urdf", "mass_kg": 0.15, "collision_hulls": 3, "mesh_count": 4,
            "source_sha256": hashes, "missing_visual_dependencies": sorted(missing), "visual_fidelity_validated": False}


def bundle_path(data_dir=None):
    value = data_dir if data_dir is not None else os.environ.get("REGRIND_DATA_DIR")
    require(value is not None and str(value).strip(), "REGRIND_DATA_DIR or --data-dir required")
    return safe_path(Path(value) / NAMESPACE)


def expected_manifest():
    return {"schema": "hand2_spatula_runtime_v1", "case": "spatula", "case_id": CASE_ID,
            "namespace": NAMESPACE, "frames": FRAMES, "control_frames": CONTROLS, "fps": 120,
            "episode_length_s": CONTROLS / 30, "source_window_30fps_closed": WINDOW30,
            "source_window_120fps_half_open": WINDOW120, "interaction_annotation_30fps_closed": ANNOTATION30,
            "historical_compliance": "spatula_noncompliant_full", "noncompliant_full_demo_fallback": True,
            "source_pins": source_pins(), "source_quality_pass": False, "source_approved_for_training": False,
            "restoration_quality_pass": True, "candidate_nonlinear_quality_pass": True,
            "robot_copy": "byte_identical_full_2229", "human_interpolation": "pinned_RHO.interpolate_demo_factor4_full",
            "human_array_sha256": HUMAN_ARRAY_SHA256, "human_wrist_quaternion_order": "xyzw",
            "robot_and_object_quaternion_order": "wxyz", "root_frame": "r_mount", "anchor_wrist_frame": "r_wrist",
            "motion_loader_wrist_target_frame": "r_mount", "physics_assumptions": physics_assumptions(),
            "reference_authorized_by_external_receipt": True, "approved_physics": False,
            "controller_validation": "pending", "restoration_proof_files": ["proof/run.json", "proof/metrics.json"]}


def validate_bundle(directory=None, *, allow_synthetic=False):
    directory = safe_path(directory if directory is not None else bundle_path())
    receipt_path = local_path(directory, "approval.json")
    require(receipt_path.is_file(), "Missing external parent receipt")
    receipt = validate_approval(json.loads(receipt_path.read_text()), allow_synthetic=allow_synthetic)
    if receipt["synthetic_test"]:
        canonical = Path(__file__).resolve().parents[4] / "data" / NAMESPACE
        require(directory.is_relative_to(safe_path(tempfile.gettempdir())) and directory != canonical, "Synthetic bundle only allowed in temporary fixtures")
    manifest = json.loads(local_path(directory, "manifest.json").read_text())
    for key, value in expected_manifest().items():
        require(manifest.get(key) == value, f"Manifest mismatch: {key}")
    require(manifest.get("approval_sha256") == sha256(receipt_path), "Receipt hash mismatch")
    model = load_model_contract()
    require(manifest.get("model") == model.provenance() and manifest.get("joint_names") == list(model.JOINT_NAMES), "Official model provenance/order mismatch")
    obj = object_payload(local_path(directory, "object"))
    require(manifest.get("object_payload") == obj, "Object provenance mismatch")
    expected_files = {"approval.json", "retargeted_120fps.h5", "demo_interp_120fps.h5", "keypoints.npy",
                      "proof/run.json", "proof/metrics.json"} | {f"object/{k}" for k in obj["source_sha256"]}
    hashes = manifest.get("output_sha256", {})
    require(set(hashes) == expected_files, "Output hash inventory differs")
    actual = set()
    for path in directory.rglob("*"):
        safe_path(path, directory)
        require(path.is_dir() or path.is_file(), "Nonregular bundle entry")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    require(actual == expected_files | {"manifest.json"}, "Unexpected bundle files")
    for relative, digest in hashes.items():
        verify_hash(local_path(directory, relative), digest)
    for relative, digest in (("retargeted_120fps.h5", H5_SHA), ("proof/run.json", RUN_SHA),
                             ("proof/metrics.json", METRICS_SHA), ("keypoints.npy", KEYPOINT_SHA)):
        verify_hash(local_path(directory, relative), digest)
    run = json.loads((directory / "proof/run.json").read_text())
    metrics = json.loads((directory / "proof/metrics.json").read_text())
    validate_quality(run, metrics)
    with h5py.File(directory / "retargeted_120fps.h5", "r") as robot, h5py.File(directory / "demo_interp_120fps.h5", "r") as human:
        validate_robot(robot, model)
        require(set(human) == set(HUMAN_SHAPES), "Unexpected human datasets")
        for key in human:
            require(isinstance(human.get(key, getlink=True), h5py.HardLink), "HDF5 links forbidden")
            item = human[key]
            require(isinstance(item, h5py.Dataset) and not item.is_virtual and item.external is None,
                    "External/virtual human storage forbidden")
        validate_arrays(human, HUMAN_SHAPES, FRAMES)
        validate_human_arrays({k: human[k][()] for k in human})
        validate_evaluation(evaluation(robot, model), metrics, "candidate")
        for a, b in (("mano_joint_coords", "mano_joint_coords"), ("object_pos", "obj_pos"), ("object_quat", "obj_quat")):
            require(np.array_equal(robot[a][()], human[b][()]), "Robot/human/object reference mismatch")
    points = np.load(directory / "keypoints.npy", allow_pickle=False)
    require(points.shape == (37, 3) and np.isfinite(points).all(), "Wrong Spatula keypoints")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    result = validate_bundle(bundle_path(args.data_dir))
    print(json.dumps({"reference_validated": True, "physics_certified": False,
                      "missing_visual_dependencies": result["object_payload"]["missing_visual_dependencies"]}))


if __name__ == "__main__":
    main()
