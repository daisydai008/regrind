"""Portable Knife staged-pilot contract; runnable before AppLauncher, no Drake.

Geometry evidence is pinned, not recomputed. Object/table support is report-only;
external approval never rewrites historical diagnostic H5 attributes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import h5py
import numpy as np

_spec = importlib.util.spec_from_file_location("knife_generic_checks", Path(__file__).with_name("hand2_case_contract.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
require, safe_path, local_path = G.require, G.safe_path, G.local_path
sha256, verify_hash, load_model_contract = G.sha256, G.verify_hash, G.load_model_contract
ROBOT_SHAPES, HUMAN_SHAPES, CONSTANT_DATASETS = G.ROBOT_SHAPES, G.HUMAN_SHAPES, G.CONSTANT_DATASETS
MODEL_PIN = "829c9aa27cb12c3a49d5b8d63c9869dd8f656702"
SOURCE = "data/tactidex/retargeted/wujihand2_diagnostic/knife_parent_weld_meshtip_table002_v1"
DEMO = "data/tactidex/regrind_demo_meshtip_hand2_right_v1/TactiDex_Cut_S0H01A0022T1_R/demo_30fps.h5"
OBJECT = "data/tactidex/assets/knife"
KEYPOINTS = "data/tactidex/keypoints/knife_100_filtered_cut.npy"
FROZEN = ".pi/skills/regrind-right-hand/references/hand2-right-eight.yaml"
RHO = "repos/regrind/scripts/retarget_hand_object.py"
PINS = {
    SOURCE + "/retargeted_120fps.h5": "77f7337d92688ca154d67dcd905c334ae96ea9b06b07214aee134cc18ba05037",
    SOURCE + "/metrics.json": "ec7197055a8309e0bd4286e4a8056607b01ebf6a0920154b111af41acd47476b",
    SOURCE + "/run.json": "58e301418ed9a9b619d7f75b84739533b21b4395609449ba637bb0cb3c9ad61c",
    DEMO: "87f70b29fc1cda8e0c2c340d6e483a9d96124ef004a9612b0aa6ac0e8fdb24d4",
    OBJECT + "/knife.urdf": "3acf5dafbdd1a7a4a2848fc3f0e6cf00d3a552eaa9dcbead72d75d63c780a610",
    KEYPOINTS: "048e8845a5e74f0af8363b79001aa329365a8e59988509612923c97420af2c55",
    FROZEN: "ac39f690cdb425982ab862dde07a0f6da9e66d04aa73fcf7af14b7c8d947324d",
    RHO: "86cc71fc8d228f3c5102ea4a5abf7b8ef31b104eb400a90d8aa71251faa546cf",
}
MAPPING_SHA = "c8b6dade2a775bc2140a55d82a606a43ee19e94245b6a80927b0930eed004cf5"
WINDOWS = ((133, 161), (192, 225), (249, 280), (297, 333), (354, 392))
CROP = slice(528, 2241)
BOUND = .001001
OBJECT_TABLE_MAX = .0053278441213758295
CASE_ID = "cut_knife_R_s0t1_sw05_tablefix"
# Logical H5 hashes include every dataset, dtype, shape, creation property and attribute.
ROBOT_SLICE_SHA = "4eb5b5819fcab0c78e5962ee69b678761fe202227b8d2d8177c24bd063d65170"
HUMAN_SLICE_SHA = "9b07b5434a09acfb5c805e23d1136852cb9b3031cb3d0b56b74cdafccd9ee092"


def approval_fields():
    return {"schema": "hand2_knife_reference_approval_v1", "case": "knife", "case_id": CASE_ID,
            "model_pin": MODEL_PIN, "source_pins": PINS,
            "window_30fps_closed": [132, 560], "window_120fps_half_open": [528, 2241],
            "table_height_m": .002, "allowed_windows_crop_30fps_closed": [list(w) for w in WINDOWS],
            "force_threshold_N": 10.0, "scope": "Knife staged pilot only"}


def validate_approval(receipt, *, allow_synthetic=False):
    require(isinstance(receipt, dict) and receipt.get("approved") is True, "External approval required")
    for key, expected in approval_fields().items():
        require(receipt.get(key) == expected, f"Approval mismatch: {key}")
    require(isinstance(receipt.get("user_decision"), str) and receipt["user_decision"].strip(), "Missing user_decision")
    require(receipt.get("synthetic_test") is False or (allow_synthetic and receipt.get("synthetic_test") is True
            and "synthetic TEST" in receipt["user_decision"]), "Synthetic TEST is not real approval")
    return receipt


def test_owner(directory, owner):
    require(isinstance(owner, tempfile.TemporaryDirectory) and safe_path(owner.name).is_dir()
            and safe_path(directory).is_relative_to(safe_path(owner.name))
            and safe_path(directory) != safe_path(owner.name), "Synthetic TEST requires caller-owned TemporaryDirectory")


def validate_quality(run, metrics):
    for record in (run, metrics):
        require(record["case"] == "knife" and record["quality_pass"] is True
                and record["approved_for_training"] is False and record["status"] == "quality_pass",
                "Historical source quality/approval changed")
    require(run["model_pin"] == MODEL_PIN and run["frames_completed"] == 3005
            and metrics["frames_120fps"] == 3005 and metrics["penetration_bound_m"] == BOUND, "Source scope mismatch")
    require(run["native_crop_120fps_half_open"] == [528, 2241]
            and run["frozen_allowed_windows_crop_30fps_closed"] == [list(w) for w in WINDOWS], "Native/window mismatch")
    require(all(metrics["validity"][k] is True for k in ("finite", "normalized_quaternions", "object_locked",
            "joint_limits", "nonlinear_self_collision", "nonlinear_object_collision", "nonlinear_hand_table_collision")), "Source validity failed")
    for kind, pairs in (("self", 310), ("object", 162), ("hand_table", 27), ("object_table", 6)):
        item = metrics["collision"][kind]
        values = np.asarray(item["per_frame_max_penetration_m"])
        distances = np.asarray(item["per_frame_min_signed_distance_m"])
        require(values.shape == distances.shape == (3005,) and np.isfinite(values).all()
                and np.isfinite(distances).all() and np.array_equal(values, np.maximum(-distances, 0))
                and item["queried_pair_count"] == pairs * 3005
                and item["max_penetration_m"] == float(values.max()), "All-frame collision proof mismatch")
        require(run["collision_filter_contract"][kind + "_candidates"] == pairs, "Collision pair count mismatch")
        if kind != "object_table":
            require(np.all(values <= BOUND) and item["frames_over_bound"] == item["count_over_bound"] == 0, "Penetration bound failed")
        else:
            require(float(values.max()) == OBJECT_TABLE_MAX and item["frames_over_bound"] == 1228,
                    "Report-only object/table evidence changed")
    exemption = run["support_exemption"]
    require(exemption["inherited_only"] is True and exemption["object_table_exempt_count"] == 6
            and exemption["hand_exempt_count"] == 0 and exemption["additional_filters"] is False,
            "Support exemption changed")
    require(metrics["object_table_policy"] == "measured_report_only_inherited_locked_object_support_exemption",
            "Object/table must remain report-only, NOT pass")
    require(run["table_geometry"] == metrics["table_geometry"] and run["table_geometry"]["height_m"] == .002,
            "Table evidence mismatch")
    for key in ("input_model_sha256", "code_sha256"):
        require(run[key] == run[key + "_after"], "Source changed during diagnostic")
    return {"method": "pinned_all_frame_geometry_evidence_not_recomputed", "metrics_sha256": PINS[SOURCE + "/metrics.json"],
            "run_sha256": PINS[SOURCE + "/run.json"], "full_frames": 3005,
            "pairs_per_frame": {"self": 310, "object": 162, "hand_table": 27, "object_table": 6},
            "bound_m": BOUND, "object_table_policy": metrics["object_table_policy"],
            "object_table_max_penetration_m": OBJECT_TABLE_MAX, "physics_certified": False}


def logical_h5(file, replacements=None):
    """Portable logical digest; replacements permit read-only expected-crop checks."""
    replacements = replacements or {}
    digest = hashlib.sha256()
    def add(value):
        a = np.asarray(value)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(json.dumps(a.tolist(), sort_keys=True).encode() if a.dtype.hasobject else a.tobytes())
    def visit(name, item):
        add(name)
        for key in sorted(item.attrs):
            add(key)
            add(item.attrs[key])
        if isinstance(item, h5py.Dataset):
            for key in ("chunks", "compression", "compression_opts", "shuffle", "fletcher32", "scaleoffset"):
                add(str(getattr(item, key)))
            add(replacements.get(name, item[()]))
    G.finite_h5(file)
    visit("/", file)
    file.visititems(visit)
    return digest.hexdigest()


def validate_robot(file, model, frames=1713):
    G.validate_arrays(file, ROBOT_SHAPES, frames)
    require(set(file) == set(ROBOT_SHAPES) | CONSTANT_DATASETS, "Robot inventory changed")
    for key, expected in {"case": "knife", "root_frame": "r_mount", "anchor_wrist_frame": "r_wrist",
                          "fps": 120, "quaternion_order": "wxyz", "quality_pass": True,
                          "approved_for_training": False, "table_height_m": .002,
                          "window_role": "full_diagnostic_NOT_training", "collision_policy": "parent-weld"}.items():
        require(file.attrs.get(key) == expected, f"Historical attribute changed: {key}")
    require(np.array_equal(file.attrs["range_120fps_closed"], [0, 3004]), "Historical range changed")
    require(hashlib.sha256(file.attrs["mapping"].encode()).hexdigest() == MAPPING_SHA
            and json.loads(file.attrs["mapping"]) == model.C.MANO_TO_ROBOT_MAPPING, "Mapping changed")
    for key, values in (("robot_joint_names", model.JOINT_NAMES), ("mano_joint_names", model.C.MANO_JOINTS),
                        ("mano_keypoint_names", model.C.MANO_TO_ROBOT_MAPPING),
                        ("robot_keypoint_link_names", model.C.MANO_TO_ROBOT_MAPPING.values())):
        require(file[key].asstr()[:].tolist() == list(values), f"Wrong named order: {key}")
    q = file["robot_joints"][()]
    for i, p in enumerate(model.PARAMETERS.values()):
        require(np.all(q[:, i] >= p["lower"] - 1e-6) and np.all(q[:, i] <= p["upper"] + 1e-6), "Joint limits")


def object_payload(directory, source_hashes, *, importer=False):
    directory = safe_path(directory)
    urdf = local_path(directory, "knife.urdf")
    verify_hash(urdf, PINS[OBJECT + "/knife.urdf"])
    raw = urdf.read_bytes()
    xml = ET.fromstring(raw)
    links = xml.findall("link")
    require(len(links) == 1 and links[0].get("name") == "bottom" and not xml.findall("joint"), "Knife rigid bottom required")
    require(float(links[0].find("inertial/mass").get("value")) == .3, "Knife mass changed")
    hulls = links[0].findall("collision/geometry/mesh")
    meshes = {e.get("filename") for e in xml.findall(".//mesh")}
    require(len(hulls) == 6 and len({e.get('filename') for e in hulls}) == 6 and len(meshes) == 7, "Six distinct hulls required")
    hashes = {"knife.urdf": PINS[OBJECT + "/knife.urdf"]}
    missing = set()
    for name in sorted(meshes):
        mesh = local_path(directory, name)
        expected = source_hashes[OBJECT + "/" + name]
        verify_hash(mesh, expected)
        hashes[name] = expected
        for line in mesh.read_text().splitlines():
            if line.startswith("mtllib "):
                material = line.split(maxsplit=1)[1]
                require(not local_path(directory, material).exists(), "Do not repair missing source material")
                missing.add(material)
    require(missing == {"3DModel.mtl"}, "Missing material state changed")
    old, new = b'filename="3DModel.obj"', b'filename="mesh_visual.obj"'
    require(raw.count(old) == 1 and xml.find("link/visual/geometry/mesh").get("filename") == "3DModel.obj", "Visual alias precondition")
    transformed = raw.replace(old, new, 1)
    alias = {"source_urdf": "object/knife.urdf", "importer_urdf": "object/knife_import.urdf",
             "source_mesh": "object/3DModel.obj", "alias_mesh": "object/mesh_visual.obj",
             "source_urdf_sha256": hashes["knife.urdf"], "importer_urdf_sha256": hashlib.sha256(transformed).hexdigest(),
             "source_mesh_sha256": hashes["3DModel.obj"], "alias_mesh_sha256": hashes["3DModel.obj"],
             "replacement": {"old": old.decode(), "new": new.decode(), "count": 1},
             "scope": "importer-only visual filename alias; inertial and six collisions byte-equivalent"}
    if importer:
        require(local_path(directory, "knife_import.urdf").read_bytes() == transformed, "Importer changed beyond visual filename")
        verify_hash(local_path(directory, "mesh_visual.obj"), hashes["3DModel.obj"])
    return {"urdf": "object/knife_import.urdf", "source_sha256": hashes, "mass_kg": .3,
            "collision_hulls": 6, "keypoint_body": "bottom", "keypoint_count": 50,
            "missing_visual_dependencies": sorted(missing), "visual_fidelity_validated": False, "importer_alias": alias}


def bundle_path(data_dir=None):
    value = data_dir if data_dir is not None else os.environ.get("REGRIND_DATA_DIR")
    require(value is not None and str(value).strip(), "REGRIND_DATA_DIR or --data-dir required")
    return safe_path(Path(value) / "hand2_right/knife_meshtip_v1")


def expected_manifest():
    return {"schema": "hand2_knife_runtime_v1", "case": "knife", "case_id": CASE_ID, "source_pins": PINS,
            "frames": 1713, "control_frames": 429, "source_full_frames": 3005, "fps": 120,
            "window_30fps_closed": [132, 560], "window_120fps_half_open": [528, 2241],
            "allowed_windows_crop_30fps_closed": [list(w) for w in WINDOWS], "force_threshold_N": 10.,
            "table_height_m": .002, "source_quality_pass": True, "source_approved_for_training": False,
            "approved_for_training": True, "approval_scope": "Knife staged pilot only", "physics_certified": False,
            "fixed_parameters": True, "augmentation": False, "speed_factor": 1.0,
            "human_interpolation": "pinned_RHO.interpolate_demo_factor4_full_then_slice",
            "robot_logical_sha256": ROBOT_SLICE_SHA, "human_logical_sha256": HUMAN_SLICE_SHA}


def validate_bundle(directory=None, *, test_directory=None):
    directory = safe_path(directory if directory is not None else bundle_path())
    if test_directory is not None:
        test_owner(directory, test_directory)
    receipt = validate_approval(json.loads(local_path(directory, "approval.json").read_text()), allow_synthetic=test_directory is not None)
    manifest = json.loads(local_path(directory, "manifest.json").read_text())
    for key, value in expected_manifest().items():
        require(manifest.get(key) == value, f"Manifest mismatch: {key}")
    for name, original in (("source_run.json", "run.json"), ("source_metrics.json", "metrics.json")):
        verify_hash(local_path(directory, name), PINS[SOURCE + "/" + original])
    run = json.loads((directory / "source_run.json").read_text())
    metrics = json.loads((directory / "source_metrics.json").read_text())
    require(manifest.get("geometry_proof") == validate_quality(run, metrics), "Geometry provenance mismatch")
    inputs = {**PINS, **run["input_model_sha256"], **run["code_sha256"]}
    require(manifest.get("verified_execution_inputs") == inputs, "Source provenance mismatch")
    model = load_model_contract()
    require(manifest.get("model") == model.provenance() and manifest.get("joint_names") == list(model.JOINT_NAMES), "Model provenance mismatch")
    obj = object_payload(directory / "object", inputs, importer=True)
    require(manifest.get("object_payload") == obj, "Object provenance mismatch")
    expected = {"approval.json", "source_run.json", "source_metrics.json", "retargeted_120fps.h5", "demo_interp_120fps.h5", "keypoints.npy"}
    expected |= {"object/" + k for k in obj["source_sha256"]} | {"object/knife_import.urdf", "object/mesh_visual.obj"}
    hashes = manifest.get("output_sha256", {})
    require(set(hashes) == expected, "Hash inventory mismatch")
    actual = set()
    for path in directory.rglob("*"):
        safe_path(path, directory)
        require(path.is_file() or path.is_dir(), "Nonregular bundle entry")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
        else:
            require(path.relative_to(directory).as_posix() in {str(Path(k).parent) for k in expected}, "Unexpected bundle directory")
    require(actual == expected | {"manifest.json"}, "Actual inventory mismatch")
    for name, digest in hashes.items():
        verify_hash(local_path(directory, name), digest)
    require(manifest.get("approval") == {"receipt_file": "approval.json", "sha256": sha256(directory / "approval.json"),
                                        "scope": receipt["scope"], "user_decision": receipt["user_decision"]}, "Approval provenance mismatch")
    verify_hash(directory / "keypoints.npy", PINS[KEYPOINTS])
    points = np.load(directory / "keypoints.npy", allow_pickle=False)
    require(points.shape == (50, 3) and np.isfinite(points).all(), "Knife requires 50 ordered keypoints")
    with h5py.File(directory / "retargeted_120fps.h5", "r") as robot, h5py.File(directory / "demo_interp_120fps.h5", "r") as human:
        validate_robot(robot, model)
        G.validate_arrays(human, HUMAN_SHAPES, 1713)
        require(logical_h5(robot) == ROBOT_SLICE_SHA and logical_h5(human) == HUMAN_SLICE_SHA, "Exact slice/metadata/schema mismatch")
        require(np.array_equal(robot["mano_joint_coords"][()], human["mano_joint_coords"][()])
                and np.array_equal(robot["object_pos"][()], human["obj_pos"][()]), "Human/object reference mismatch")
        require(np.allclose(np.abs(np.sum(robot["object_quat"][()] * human["obj_quat"][()], axis=1)), 1, rtol=0, atol=1e-6), "Object rotation mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    validate_bundle(bundle_path(args.data_dir))
    print(json.dumps({"reference_validated": True, "physics_certified": False, "visual_fidelity_validated": False}))


if __name__ == "__main__":
    main()
