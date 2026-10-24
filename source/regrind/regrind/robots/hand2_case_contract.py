"""Portable, read-only Hand2 case reference checks; no Drake or task registry imports.

Run directly BEFORE AppLauncher. Reference approval is not physics certification.
Historical H5 quality/approval/range attributes are deliberately not rewritten.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import h5py
import numpy as np

MODEL_PIN = "829c9aa27cb12c3a49d5b8d63c9869dd8f656702"
FROZEN_REL = ".pi/skills/regrind-right-hand/references/hand2-right-eight.yaml"
FROZEN_SHA = "ac39f690cdb425982ab862dde07a0f6da9e66d04aa73fcf7af14b7c8d947324d"
RHO_REL = "repos/regrind/scripts/retarget_hand_object.py"
RHO_SHA = "86cc71fc8d228f3c5102ea4a5abf7b8ef31b104eb400a90d8aa71251faa546cf"
CASES = {
    "kettleblack": {
        "label": "KettleBlack", "id": "pour_kettleblack_R_s0t1_sw05",
        "take": "TactiDex_Pour_S0H01A0018T1_R", "frames30": 924, "full_frames": 3693,
        "window30": [241, 801], "window120": [964, 3205], "frames": 2241, "control_frames": 561,
        "keypoints": "data/tactidex/keypoints/kettleblack_100_filtered.npy", "keypoint_count": 8,
        "keypoint_sha": "ca307b7fdbbfd755a9a58aaa84c2b84cd1695f3077b52c5b1910fa8596d044dc",
        "demo_sha": "36f3f8844db26563d4322c631dbd428e38e357b456f9e149f8a2ccf119ece9c3",
        "h5_sha": "8729f85f5042cb7270f3dcdcc4853d28cf3ef6d4679cd22f7c9ebf1ed98f1354",
        "run_sha": "a4a711a31f2b11345c65128690bfdc93bdad99c8ae7da4d32f9a9598e6215456",
        "metrics_sha": "65e490e7e71044095bf6eda514cc210e033afced49147b4be64878d329be79dd",
        "original_sha": "eff956bd9fb3103d725c4b06682b697a930dd5ded9b91fe19368358812601b68",
        "object_dir": "data/tactidex/assets/KettleBlack", "mass": 0.5,
        "urdf_sha": "899eae84f0eef595315648c09d6266570e34e2db974bf481aac9d8306b30bae3",
        "object_payload_sha": "7eb972100ca3ac43234f466b1e9bafaf60df3323c051fe51bfd4bec8412cfadb",
        "missing_textures": ["KettleBlack.jpg"], "bad_count": 267,
        "lower_bound": [-0.414, -0.159, 0.0], "upper_bound": [-0.039, 0.199, 0.526],
    },
    "spoon": {
        "label": "Spoon", "id": "scoop_spoon_R_s0t1_sw05",
        "take": "TactiDex_Scoop_S0H01A0037T1_R", "frames30": 819, "full_frames": 3273,
        "window30": [132, 659], "window120": [528, 2637], "frames": 2109, "control_frames": 528,
        "keypoints": "data/tactidex/keypoints/spoon_100_filtered_scoop.npy", "keypoint_count": 37,
        "keypoint_sha": "49961811d79e4124b65c19890a5b59fa23b4373a1ce410207478d94a0856d5c5",
        "demo_sha": "163426c17293427a420bd6bc83e3c6698cad8a1286d12ce530ae8759a6f4b402",
        "h5_sha": "ebe39597c13b8d4ad8b63f2db8fc1f90823b98f4c252f25256a6cb4440cf959b",
        "run_sha": "f2aaa3cfece33e16d9c640e639ec0f22dbb88a27745b9fafe3ead46e1135aa2a",
        "metrics_sha": "3aa4f3d6596c00c0d1a30e2e65370fdda8fe7974daf380a2a847a67e0674bf22",
        "original_sha": "ce968e4ad510820c84549b1d496fb9542e94adb28df7f2c5453d043c7bc7284b",
        "object_dir": "data/tactidex/assets/spoon", "mass": 0.05,
        "urdf_sha": "f16fa4897cd138c6c5c0f672051f0ae122f6b5c4323f30af860e165457e3ae4d",
        "object_payload_sha": "152e3c2f9885b3199bce1429443bb0a1890d2dea94331f943b1800a598fe3927",
        "missing_textures": [], "bad_count": 20,
        "lower_bound": [-0.439, -0.153, 0.0], "upper_bound": [-0.055, 0.413, 0.41],
    },
}
ROBOT_SHAPES = {"robot_joints": (20,), "robot_pos": (3,), "robot_quat": (4,),
                "object_pos": (3,), "object_quat": (4,), "anchor_wrist_pos": (3,),
                "anchor_wrist_quat": (4,), "robot_keypoints": (21, 3), "mano_joint_coords": (21, 3)}
HUMAN_SHAPES = {"mano_joint_coords": (21, 3), "obj_pos": (3,), "obj_quat": (4,),
                "wrist_pos": (3,), "wrist_quat": (4,)}
CONSTANT_DATASETS = {"mano_joint_names", "mano_keypoint_names", "robot_joint_names", "robot_keypoint_link_names"}
BOUND = 0.001001


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(path, base=None):
    raw = Path(path).expanduser()
    require(".." not in raw.parts, f"Path traversal: {path}")
    path = Path(os.path.abspath(raw))
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"Symlink rejected: {path}")
    if base is not None:
        require(path.is_relative_to(safe_path(base)), f"Path escape: {path}")
    return path


def local_path(base, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
            and ":" not in relative, "Nonlocal payload path")
    return safe_path(Path(base) / relative, base)


def sha256(path):
    return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def verify_hash(path, digest):
    require(sha256(path) == digest, f"Pinned hash mismatch: {path}")


def load_model_contract():
    spec = importlib.util.spec_from_file_location("case_hand2_model", Path(__file__).with_name("hand2_right_contract.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_prefix(name):
    require(name in CASES, "Unsupported case")
    return f"data/tactidex/retargeted/wujihand2_diagnostic/{name}_parent_weld_meshtip_feasible_v1"


def demo_relative(name):
    return f"data/tactidex/regrind_demo_meshtip_hand2_right_v1/{CASES[name]['take']}/demo_30fps.h5"


def source_pins(name):
    c = CASES[name]
    return {FROZEN_REL: FROZEN_SHA, RHO_REL: RHO_SHA, demo_relative(name): c["demo_sha"],
            c["keypoints"]: c["keypoint_sha"], f"{c['object_dir']}/{name}.urdf": c["urdf_sha"],
            **{f"{source_prefix(name)}/{file}": c[key] for file, key in
               (("retargeted_120fps.h5", "h5_sha"), ("run.json", "run_sha"), ("metrics.json", "metrics_sha"))}}


def approval_fields(name):
    c = CASES[name]
    return {"schema": "hand2_case_reference_approval_v1", "case": name, "case_id": c["id"],
            "model_pin": MODEL_PIN, "full_h5_sha256": c["h5_sha"], "source_demo_sha256": c["demo_sha"],
            "window_30fps_closed": c["window30"], "scope": f"{c['label']} pilot only"}


def validate_approval(receipt, name, *, allow_synthetic=False):
    require(isinstance(receipt, dict) and receipt.get("approved") is True, "Explicit external approval required")
    for key, value in approval_fields(name).items():
        require(receipt.get(key) == value, f"Approval mismatch: {key}")
    require(isinstance(receipt.get("user_decision"), str) and receipt["user_decision"].strip(), "Missing user_decision")
    require(receipt.get("synthetic_test") is False or
            (allow_synthetic and receipt.get("synthetic_test") is True and "synthetic TEST" in receipt["user_decision"]),
            "Synthetic TEST is not real approval; synthetic_test=false required")
    return receipt


def finite_h5(file):
    def visit(group):
        for key in group:
            require(isinstance(group.get(key, getlink=True), h5py.HardLink), "HDF5 links forbidden")
            item = group[key]
            if isinstance(item, h5py.Group):
                visit(item)
            elif np.issubdtype(item.dtype, np.number):
                require(np.isfinite(item[()]).all(), f"Nonfinite H5: {item.name}")
    visit(file)


def validate_arrays(file, shapes, frames):
    finite_h5(file)
    for key, tail in shapes.items():
        require(file[key].shape == (frames, *tail), f"Wrong shape: {key}")
        if "quat" in key:
            require(np.max(np.abs(np.linalg.norm(file[key][()], axis=-1) - 1)) < 1e-6, f"Nonunit quaternion: {key}")


def validate_robot(file, name, model, frames=None):
    c = CASES[name]
    validate_arrays(file, ROBOT_SHAPES, c["full_frames"] if frames is None else frames)
    require(set(file) == set(ROBOT_SHAPES) | CONSTANT_DATASETS, "Unexpected robot datasets")
    attrs = {"case": name, "root_frame": "r_mount", "anchor_wrist_frame": "r_wrist", "fps": 120,
             "robot_name": "wujihand2_right_diagnostic_parent_weld", "quaternion_order": "wxyz",
             "collision_policy": "parent-weld", "quality_pass": False, "approved_for_training": False,
             "restoration_quality_pass": True, "restoration_source_h5_sha256": c["original_sha"],
             "restoration_stage": "minimum_mixed_coordinate_L2_feasibility_projection",
             "window_role": "full_diagnostic_NOT_training"}
    for key, expected in attrs.items():
        require(file.attrs.get(key) == expected, f"Historical H5 metadata changed: {key}")
    require(np.array_equal(file.attrs.get("range_120fps_closed"), [0, c["full_frames"] - 1]), "Historical full range changed")
    require(json.loads(file.attrs["mapping"]) == model.C.MANO_TO_ROBOT_MAPPING, "Wrong mapping metadata")
    for key, expected in (("robot_joint_names", model.JOINT_NAMES), ("mano_joint_names", model.C.MANO_JOINTS),
                          ("mano_keypoint_names", model.C.MANO_TO_ROBOT_MAPPING),
                          ("robot_keypoint_link_names", model.C.MANO_TO_ROBOT_MAPPING.values())):
        require(file[key].asstr()[:].tolist() == list(expected), f"Wrong official order: {key}")
    q = file["robot_joints"][()]
    for i, params in enumerate(model.PARAMETERS.values()):
        require(np.all(q[:, i] >= params["lower"] - 1e-6) and np.all(q[:, i] <= params["upper"] + 1e-6), "Joint limits")


def object_payload(directory, name):
    """Validate byte-pinned local URDF, all 11 meshes, MTLs and present textures."""
    c = CASES[name]
    urdf = local_path(directory, f"{name}.urdf")
    verify_hash(urdf, c["urdf_sha"])
    root = ET.parse(urdf).getroot()
    links = root.findall("link")
    require(len(links) == 1 and links[0].get("name") == "bottom" and not root.findall("joint"), "Object must be rigid bottom")
    require(float(links[0].find("inertial/mass").get("value")) == c["mass"], "Actual source URDF mass changed")
    collisions = links[0].findall("collision/geometry/mesh")
    meshes = {m.get("filename") for m in root.findall(".//mesh")}
    require(len(collisions) == 10 and len(meshes) == 11 and len({m.get('filename') for m in collisions}) == 10, "Expected ten distinct hulls and visual OBJ")
    hashes, missing = {}, set()
    for relative in sorted(meshes):
        mesh = local_path(directory, relative)
        hashes[relative] = sha256(mesh)
        for line in mesh.read_text().splitlines():
            if line.startswith("mtllib "):
                material_rel = line.split(maxsplit=1)[1]
                material = local_path(directory, material_rel)
                hashes[material_rel] = sha256(material)
                for entry in material.read_text().splitlines():
                    if entry.startswith("map_"):
                        rel = entry.split(maxsplit=1)[1]
                        texture = local_path(directory, rel)
                        if texture.exists():
                            hashes[rel] = sha256(texture)
                        else:
                            missing.add(rel)
    require(sorted(missing) == c["missing_textures"], "Unexpected missing texture state; no repair/auto-ignore")
    source_hashes = {f"{c['object_dir']}/{k}": v for k, v in hashes.items()}
    require(hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest() == c["object_payload_sha"], "Object payload pin mismatch")
    hashes[f"{name}.urdf"] = c["urdf_sha"]
    return {"urdf": f"object/{name}.urdf", "mass_kg": c["mass"], "collision_hulls": 10,
            "mesh_count": 11, "source_sha256": hashes, "missing_visual_textures": sorted(missing),
            "visual_fidelity_validated": False}


def bundle_path(name, data_dir=None):
    require(name in CASES, "Unsupported case")
    value = data_dir if data_dir is not None else os.environ.get("REGRIND_DATA_DIR")
    require(value is not None and str(value).strip(), "REGRIND_DATA_DIR or --data-dir required")
    return safe_path(Path(value) / "hand2_right" / f"{name}_meshtip_v1")


def expected_manifest(name):
    c = CASES[name]
    return {"schema": "hand2_case_runtime_v1", "case": name, "case_id": c["id"],
            "frames": c["frames"], "source_full_frames": c["full_frames"], "control_frames": c["control_frames"],
            "source_window_30fps_closed": c["window30"], "source_window_120fps_half_open": c["window120"],
            "fps": 120, "source_pins": source_pins(name), "source_quality_pass": False,
            "restoration_quality_pass": True, "source_approved_for_training": False,
            "approved_for_training": True, "approved_physics": False, "controller_validation": "pending",
            "fixed_parameters": True, "augmentation": False, "speed_factor": 1.0,
            "root_frame": "r_mount", "anchor_wrist_frame": "r_wrist", "motion_loader_wrist_target_frame": "r_mount",
            "human_interpolation": "pinned_RHO.interpolate_demo_factor4_full_then_slice",
            "expected_shapes": {"robot": {k: [c['frames'], *v] for k, v in ROBOT_SHAPES.items()},
                                "human": {k: [c['frames'], *v] for k, v in HUMAN_SHAPES.items()}}}


def validate_bundle(directory=None, name=None, *, allow_synthetic=False):
    require(name in CASES, "Explicit case required")
    directory = safe_path(directory if directory is not None else bundle_path(name))
    receipt_path = local_path(directory, "approval.json")
    require(receipt_path.is_file(), "Missing external approval; training forbidden")
    receipt = validate_approval(json.loads(receipt_path.read_text()), name, allow_synthetic=allow_synthetic)
    manifest = json.loads(local_path(directory, "manifest.json").read_text())
    for key, expected in expected_manifest(name).items():
        require(manifest.get(key) == expected, f"Manifest mismatch: {key}")
    require(manifest.get("approval") == {"receipt_file": "approval.json", "sha256": sha256(receipt_path),
                                       "scope": receipt["scope"], "user_decision": receipt["user_decision"]}, "Approval provenance mismatch")
    model = load_model_contract()
    require(manifest.get("model") == model.provenance(), "Model provenance mismatch")
    require(manifest.get("joint_names") == list(model.JOINT_NAMES), "Joint names mismatch")
    verified = manifest.get("verified_execution_inputs", {})
    require(isinstance(verified, dict) and all(verified.get(k) == v for k, v in source_pins(name).items()),
            "Source execution provenance is not bound to pinned inputs")
    payload = object_payload(local_path(directory, "object"), name)
    require(manifest.get("object_payload") == payload, "Object provenance mismatch")
    expected_files = {"approval.json", "retargeted_120fps.h5", "demo_interp_120fps.h5", "keypoints.npy"} | {f"object/{k}" for k in payload["source_sha256"]}
    hashes = manifest.get("output_sha256", {})
    require(set(hashes) == expected_files, "Output hash inventory differs")
    actual_files = set()
    for path in directory.rglob("*"):
        safe_path(path, directory)
        if path.is_file():
            actual_files.add(path.relative_to(directory).as_posix())
    require(actual_files == expected_files | {"manifest.json"}, "Unexpected bundle files")
    for relative, digest in hashes.items():
        verify_hash(local_path(directory, relative), digest)
    verify_hash(directory / "keypoints.npy", CASES[name]["keypoint_sha"])
    with h5py.File(directory / "retargeted_120fps.h5", "r") as robot, h5py.File(directory / "demo_interp_120fps.h5", "r") as human:
        validate_robot(robot, name, model, CASES[name]["frames"])
        require(set(human) == set(HUMAN_SHAPES), "Unexpected human datasets")
        validate_arrays(human, HUMAN_SHAPES, CASES[name]["frames"])
        for a, b in (("mano_joint_coords", "mano_joint_coords"), ("object_pos", "obj_pos"), ("object_quat", "obj_quat")):
            require(np.array_equal(robot[a][()], human[b][()]), f"Human/object reference mismatch: {a}")
    points = np.load(directory / "keypoints.npy", allow_pickle=False)
    require(points.shape == (CASES[name]["keypoint_count"], 3) and np.isfinite(points).all(), "Wrong case keypoints")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    result = validate_bundle(bundle_path(args.case, args.data_dir), args.case)
    print(json.dumps({"reference_validated": True, "physics_certified": False,
                      "missing_visual_textures": result["object_payload"]["missing_visual_textures"]}))


if __name__ == "__main__":
    main()
