"""Hand2.2 RIGHT diagnostic resources, NOT production/calibration/Isaac support.

Body origins are used directly, including actual tips (not distal-origin pads).
The native floating r_mount and vendor joint ordering are never rewritten.
"""
import hashlib
import json
from pathlib import Path

ROBOT_NAME = "wujihand2_right_diagnostic"
ROBOT_DOF = 20
MODEL_PIN = "829c9aa27cb12c3a49d5b8d63c9869dd8f656702"
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets/hand2_2_right"
ROBOT_URDF = ASSET_DIR / "vendor_drake.urdf"
MANIFEST_SHA256 = "e7153478d1d955a39cebe480ffd48e99ab2c81f9d8113a975d6c41667ff03d66"
MJCF_SOURCE = "hand2/hand2_2/body/mjcf/right.xml"
MJCF_SHA256 = "c11ab34b0fd0c1d520d108ffdefe45bfc05711bb2e97946a6bda96888cf02847"
MANO_JOINTS = [str(i) for i in range(21)]
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
MANO_TO_ROBOT_MAPPING = {"0": "r_wrist"}
for _finger, _ids in zip(FINGERS, ((13, 14, 15, 16), (1, 2, 3, 17),
                                      (4, 5, 6, 18), (10, 11, 12, 19), (7, 8, 9, 20))):
    for _id, _suffix in zip(_ids, ("proximal_abd", "middle", "distal", "tip")):
        MANO_TO_ROBOT_MAPPING[str(_id)] = f"r_{_finger}_{_suffix}"
JOINT_NAMES = [f"r_{finger}_{suffix}" for finger in FINGERS
               for suffix in (("cmc_flex", "cmc_abd", "mcp", "ip") if finger == "thumb"
                              else ("mcp_flex", "mcp_abd", "pip", "dip"))]
MCP_LINKS = [f"r_{finger}_proximal" for finger in FINGERS[1:]]
MCP_IDS = [1, 4, 10, 7]
# Exactly the source MJCF's 15 entries; not inferred from adjacency or contacts.
EXPECTED_EXCLUSIONS = [(f"r_{finger}_{suffix}", "r_wrist") for finger in FINGERS
                       for suffix in ("proximal", "proximal_abd")]
EXPECTED_EXCLUSIONS += [(f"r_{finger}_distal", f"r_{finger}_tip_sensor_frame")
                        for finger in ("index_finger", "middle_finger", "pinky", "ring_finger", "thumb")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_asset(asset_dir=ASSET_DIR):
    """Reject modified manifests or generated payloads, without relabeling the asset."""
    asset_dir = Path(asset_dir)
    path = asset_dir / "manifest.json"
    if sha256(path) != MANIFEST_SHA256:
        raise ValueError("Hand2 diagnostic manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if (manifest["model_pin"] != MODEL_PIN or manifest["model"] != "Hand2.2 RIGHT"
            or manifest["status"] != "preparatory"
            or manifest["approved_for_retargeting"] or manifest["approved_for_training"]
            or manifest["source_files_sha256"][MJCF_SOURCE] != MJCF_SHA256):
        raise ValueError("Hand2 preparatory model/source contract mismatch")
    pairs = [(p["body1"], p["body2"]) for p in manifest["vendor_mjcf_exclusions"]["pairs"]]
    if pairs != EXPECTED_EXCLUSIONS or len(set(pairs)) != 15:
        raise ValueError("Hand2 source exclusions mismatch")
    for relative, expected in manifest["generated_files_sha256"].items():
        path = asset_dir / relative
        if not path.resolve().is_relative_to(asset_dir.resolve()) or sha256(path) != expected:
            raise ValueError(f"Hand2 generated payload hash mismatch: {relative}")
    return manifest


def setup_collision_exclusions(plant, scene_graph):
    """Apply only the pinned 15 source pairs; leave Drake default filters intact."""
    from pydrake.geometry import CollisionFilterDeclaration, GeometrySet

    manifest = validate_asset()
    declaration = CollisionFilterDeclaration()
    for pair in manifest["vendor_mjcf_exclusions"]["pairs"]:
        bodies = [plant.GetBodyByName(pair[k]) for k in ("body1", "body2")]
        geometries = [plant.GetCollisionGeometriesForBody(body) for body in bodies]
        if not all(geometries):
            raise ValueError(f"Missing source exclusion collision geometry: {pair}")
        declaration.ExcludeBetween(*(GeometrySet(ids) for ids in geometries))
    scene_graph.collision_filter_manager().Apply(declaration)


def assert_layout(plant):
    """Fail before any packed-q slicing: robot 27, rigid object 7, total 34."""
    from pydrake.multibody.tree import JointIndex

    if plant.num_positions() != 34:
        raise ValueError("Expected robot 27 + rigid object 7 positions")
    mount = plant.GetBodyByName("r_mount")
    obj = plant.GetBodyByName("bottom")
    if (plant.num_positions(mount.model_instance()) != 27
            or plant.num_positions(obj.model_instance()) != 7
            or not mount.is_floating_base_body() or not mount.has_quaternion_dofs()
            or mount.floating_positions_start() != 0
            or not obj.is_floating_base_body() or not obj.has_quaternion_dofs()
            or obj.floating_positions_start() != 27):
        raise ValueError("Expected native r_mount[0:7], joints[7:27], object[27:34]")
    actual = sorted((plant.get_joint(JointIndex(i)).position_start(),
                     plant.get_joint(JointIndex(i)).name())
                    for i in range(plant.num_joints())
                    if plant.get_joint(JointIndex(i)).num_positions() == 1)
    if actual != list(zip(range(7, 27), JOINT_NAMES)):
        raise ValueError(f"Unexpected actuated q ordering: {actual}")
    expected = ([f"r_mount_{s}" for s in ("qw", "qx", "qy", "qz", "x", "y", "z")]
                + JOINT_NAMES + [f"bottom_{s}" for s in ("qw", "qx", "qy", "qz", "x", "y", "z")])
    if list(plant.GetPositionNames(False, False)) != expected:
        raise ValueError("Unexpected quaternion/joint position names")
    for name in MANO_TO_ROBOT_MAPPING.values():
        plant.GetBodyByName(name)
