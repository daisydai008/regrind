"""Sim-free MeshTip reference validation; source quality is NOT user approval.

Before any training AppLauncher, run this file directly with --data-dir (or
REGRIND_DATA_DIR). The shared training entry point is deliberately unchanged.
Config initialization repeats validation before constructing the environment.
No raw demo, workspace publisher, vendor checkout, or simulator is needed here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np

NAMESPACE = Path("hand2_right/bowl_meshtip_v1")
MODEL_PIN = "829c9aa27cb12c3a49d5b8d63c9869dd8f656702"
FULL_H5_SHA = "06f9b6748ca74b36ea970723e60343851ea55553d17e020bec17089861423d88"
DEMO_SHA = "4293ab9d3558bdf6da175bc2a1e25f9940b71e7ce41a3111f1cba90cccd8bbb3"
SOURCE_PREFIX = "data/tactidex/retargeted/wujihand2_diagnostic/bowl_parent_weld_meshtip_v1/"
DEMO_REL = "data/tactidex/regrind_demo_meshtip_test/TactiDex_Drink_S0H01A0007T1_R/demo_30fps.h5"
KEYPOINT_REL = "data/tactidex/keypoints/bowl_100_filtered_drink.npy"
RHO_REL = "repos/regrind/scripts/retarget_hand_object.py"
SOURCE_PINS = {
    SOURCE_PREFIX + "retargeted_120fps.h5": FULL_H5_SHA,
    SOURCE_PREFIX + "run.json": "8babc1c45962870284d8c7c740ee1895ed3df4972f7df5f6b4d403eed5d83b34",
    SOURCE_PREFIX + "metrics.json": "7a63c80b6d632095efb3793858195c5a42bf25e5d34e1301709b0a53595591c8",
    DEMO_REL: DEMO_SHA,
    KEYPOINT_REL: "16dcaa62c4c84e79a79a7b88a05aabfe348a05bed586f4fb6c423ce88071fe3c",
    RHO_REL: "86cc71fc8d228f3c5102ea4a5abf7b8ef31b104eb400a90d8aa71251faa546cf",
}
WINDOW = slice(564, 2549)
FRAMES, FULL_FRAMES = 1985, 3037
HUMAN_DATASETS = {"mano_joint_coords", "obj_pos", "obj_quat", "wrist_pos", "wrist_quat"}
CONSTANT_DATASETS = {"mano_joint_names", "mano_keypoint_names", "robot_joint_names", "robot_keypoint_link_names"}
OUTPUT_FILES = {"retargeted_120fps.h5", "demo_interp_120fps.h5", "keypoints.npy", "approval.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hash(path, digest):
    require(sha256(path) == digest, f"Pinned hash mismatch: {path}")


def load_model_contract():
    spec = importlib.util.spec_from_file_location(
        "meshtip_right_contract", Path(__file__).with_name("hand2_right_contract.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_approval(receipt, *, allow_synthetic=False):
    require(isinstance(receipt, dict), "Missing explicit user approval receipt")
    expected = {"schema": "hand2_reference_approval_v1", "case": "Bowl", "model_pin": MODEL_PIN,
                "full_h5_sha256": FULL_H5_SHA, "source_demo_sha256": DEMO_SHA,
                "window_30fps_closed": [141, 637], "scope": "Bowl pilot only"}
    require(receipt.get("approved") is True, "Missing explicit user approval")
    for key, value in expected.items():
        require(receipt.get(key) == value, f"Approval mismatch: {key}")
    quote = receipt.get("user_decision")
    require(isinstance(quote, str) and bool(quote.strip()), "Explicit user_decision text required; never auto-generate")
    require(receipt.get("synthetic_test", False) is False or allow_synthetic,
            "Synthetic TEST receipt is not training approval")
    return receipt


def finite_h5(file):
    def check(name, value):
        if isinstance(value, h5py.Dataset) and np.issubdtype(value.dtype, np.number):
            require(np.isfinite(value[()]).all(), f"Nonfinite dataset: {name}")
    file.visititems(check)


def names(dataset):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in dataset[()]]


def validate_robot(file, model, frames=FULL_FRAMES):
    finite_h5(file)
    for key, value in {"robot_name": "wujihand2_right_diagnostic_parent_weld", "root_frame": "r_mount",
                       "anchor_wrist_frame": "r_wrist", "collision_policy": "parent-weld",
                       "quaternion_order": "wxyz", "fps": 120}.items():
        require(file.attrs.get(key) == value, f"Wrong robot metadata: {key}")
    # These are immutable historical source attributes, not runtime authorization.
    require(file.attrs.get("approved_for_training") == False, "Historical approval metadata changed")
    require(np.array_equal(file.attrs.get("range_120fps_closed"), [0, 3036]), "Historical range changed")
    require(names(file["robot_joint_names"]) == list(model.JOINT_NAMES), "Wrong joint order")
    require(names(file["robot_keypoint_link_names"]) == list(model.C.MANO_TO_ROBOT_MAPPING.values()), "Wrong keypoint order")
    shapes = {"robot_joints": (20,), "robot_pos": (3,), "robot_quat": (4,), "object_pos": (3,),
              "object_quat": (4,), "anchor_wrist_pos": (3,), "anchor_wrist_quat": (4,),
              "robot_keypoints": (21, 3), "mano_joint_coords": (21, 3)}
    for key, tail in shapes.items():
        require(file[key].shape == (frames, *tail), f"Wrong frame/shape: {key}")
    for key in ("robot_quat", "object_quat", "anchor_wrist_quat"):
        require(np.max(np.abs(np.linalg.norm(file[key][()], axis=-1) - 1)) < 1e-6, f"Invalid quaternion: {key}")
    q = file["robot_joints"][()]
    for i, values in enumerate(model.PARAMETERS.values()):
        require(np.all(q[:, i] >= values["lower"] - 1e-6) and np.all(q[:, i] <= values["upper"] + 1e-6), "Joint limits")


def validate_human(file):
    finite_h5(file)
    require(set(file) == HUMAN_DATASETS, "Expected exactly five human-only datasets")
    for key in HUMAN_DATASETS:
        tail = (21, 3) if key == "mano_joint_coords" else ((4,) if "quat" in key else (3,))
        require(file[key].shape == (FRAMES, *tail), f"Wrong human shape: {key}")
        if "quat" in key:
            require(np.max(np.abs(np.linalg.norm(file[key][()], axis=-1) - 1)) < 1e-6, f"Invalid human quaternion: {key}")


def bundle_path(data_dir=None):
    value = data_dir if data_dir is not None else os.environ.get("REGRIND_DATA_DIR")
    require(value is not None and bool(str(value).strip()), "REGRIND_DATA_DIR or --data-dir required")
    return Path(value) / NAMESPACE


def validate_bundle(directory=None, *, allow_synthetic=False):
    directory = Path(directory) if directory is not None else bundle_path()
    # Approval is checked first, even before model hashes or H5 reads.
    require((directory / "approval.json").is_file(), "Missing explicit user approval receipt; training forbidden")
    receipt = validate_approval(json.loads((directory / "approval.json").read_text()), allow_synthetic=allow_synthetic)
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest.get("schema") == "hand2_bowl_meshtip_runtime_v1", "Wrong runtime schema")
    require(manifest.get("approved_for_training") is True, "Runtime reference not approved")
    require(manifest.get("approval") == {"receipt_file": "approval.json", "source_sha256": sha256(directory / "approval.json"),
                                      "user_decision": receipt["user_decision"], "scope": "Bowl pilot only"}, "Approval provenance mismatch")
    expected = {"frames": FRAMES, "source_full_frames": FULL_FRAMES, "control_frames": 497, "fps": 120,
                "source_window_120fps_half_open": [564, 2549], "source_window_30fps_closed": [141, 637],
                "augmentation": False, "speed_factor": 1.0, "fixed_parameters": True,
                "source_quality_pass": True, "source_approved_for_training": False,
                "approved_physics": False, "controller_validation": "pending",
                "root_frame": "r_mount", "anchor_wrist_frame": "r_wrist",
                "motion_loader_wrist_target_frame": "r_mount", "source_pins": SOURCE_PINS}
    for key, value in expected.items():
        require(manifest.get(key) == value, f"Manifest mismatch: {key}")
    model = load_model_contract()
    require(manifest.get("model") == model.provenance(), "Model contract changed")
    require(manifest.get("joint_names") == list(model.JOINT_NAMES), "Manifest joint order mismatch")
    hashes = manifest.get("output_sha256", {})
    require(set(hashes) == OUTPUT_FILES, "Missing runtime files")
    require(hashes["keypoints.npy"] == SOURCE_PINS[KEYPOINT_REL], "Wrong Bowl keypoints")
    for name, digest in hashes.items():
        verify_hash(directory / name, digest)
    with h5py.File(directory / "retargeted_120fps.h5") as file:
        validate_robot(file, model, FRAMES)
    with h5py.File(directory / "demo_interp_120fps.h5") as file:
        validate_human(file)
    points = np.load(directory / "keypoints.npy", allow_pickle=False)
    require(points.shape == (41, 3) and np.isfinite(points).all(), "Expected 41 finite Bowl keypoints")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Portable runtime data root; defaults to REGRIND_DATA_DIR")
    args = parser.parse_args()
    validate_bundle(bundle_path(args.data_dir))
    print("Approved Bowl pilot reference validated; dynamics/controller calibration NOT certified.")


if __name__ == "__main__":
    main()
