"""Pinned, uncalibrated vendor parameters for isolated Hand2 LEFT validation.

No simulator imports, vendor-checkout dependency, or filesystem writes on import.
MIT provenance remains in the existing asset manifest. No training approval.
"""
import hashlib
import importlib.util
import json
import itertools
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

# Load the existing sim-free constants without executing regrind's task registry.
_CONSTANTS_PATH = Path(__file__).resolve().parents[1] / "retargeting/wujihand2_left_diagnostic_constants.py"
_spec = importlib.util.spec_from_file_location("hand2_diagnostic_constants", _CONSTANTS_PATH)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)
JOINT_NAMES = tuple(C.JOINT_NAMES)
SOURCE_URDF = "hand2/hand2_2/body/urdf/left.urdf"
SOURCE_URDF_SHA256 = "1980c678724637c7241d2cf68eeafc74011864718f8723e5ba4ddcc99724186a"
ASSET_URDF_SHA256 = "7998fd0d5b210d5a9e54d791757e46526da6b85c3222e1f373977b3ad74774ce"
PARAMETER_STATUS = "uncalibrated_vendor_parameters"
POSITION_TOLERANCE_M = 1e-5
ROTATION_TOLERANCE_RAD = 1e-5
# Effective MJCF armature: default .0002, six explicit .0005 overrides.
PARAMETERS = {
    'l_thumb_cmc_flex': {'kp': 0.40844710645048043, 'kv': 0.020882010257063675, 'effort': 0.6, 'velocity': 8.587, 'lower': -1.187, 'upper': 1.291, 'armature': 0.0005},
    'l_thumb_cmc_abd': {'kp': 0.6858601063643346, 'kv': 0.030610939996373314, 'effort': 0.6, 'velocity': 11.1, 'lower': -1.484, 'upper': 0.698, 'armature': 0.0005},
    'l_thumb_mcp': {'kp': 0.2391112196099482, 'kv': 0.010181475050560962, 'effort': 0.3, 'velocity': 12.86, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0002},
    'l_thumb_ip': {'kp': 0.20736128319711747, 'kv': 0.00909698844675045, 'effort': 0.3, 'velocity': 13.5, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0002},
    'l_index_finger_mcp_flex': {'kp': 0.37352218155860073, 'kv': 0.01882274330029718, 'effort': 2.0, 'velocity': 8.203, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0005},
    'l_index_finger_mcp_abd': {'kp': 0.45592448909027794, 'kv': 0.019798167597016643, 'effort': 0.2, 'velocity': 8.11, 'lower': -0.698, 'upper': 0.698, 'armature': 0.0002},
    'l_index_finger_pip': {'kp': 0.24368366649522863, 'kv': 0.010477031953162727, 'effort': 0.3, 'velocity': 12.86, 'lower': -1.047, 'upper': 2.094, 'armature': 0.0002},
    'l_index_finger_dip': {'kp': 0.18026971340925335, 'kv': 0.008240212147903584, 'effort': 0.3, 'velocity': 13.5, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0002},
    'l_middle_finger_mcp_flex': {'kp': 0.3687093483646485, 'kv': 0.01848622487024593, 'effort': 2.0, 'velocity': 8.203, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0005},
    'l_middle_finger_mcp_abd': {'kp': 0.4164253443634641, 'kv': 0.018032947229953678, 'effort': 0.2, 'velocity': 8.11, 'lower': -0.698, 'upper': 0.698, 'armature': 0.0002},
    'l_middle_finger_pip': {'kp': 0.22218607502059182, 'kv': 0.009592200014076666, 'effort': 0.3, 'velocity': 12.86, 'lower': -1.047, 'upper': 2.094, 'armature': 0.0002},
    'l_middle_finger_dip': {'kp': 0.19427606072023446, 'kv': 0.009152994605972402, 'effort': 0.3, 'velocity': 13.5, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0002},
    'l_ring_finger_mcp_flex': {'kp': 0.35718151495111794, 'kv': 0.018376606800780005, 'effort': 2.0, 'velocity': 8.203, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0005},
    'l_ring_finger_mcp_abd': {'kp': 0.42977315313086895, 'kv': 0.01867700966212433, 'effort': 0.2, 'velocity': 8.11, 'lower': -0.698, 'upper': 0.698, 'armature': 0.0002},
    'l_ring_finger_pip': {'kp': 0.24930151196247122, 'kv': 0.01059512121009555, 'effort': 0.3, 'velocity': 12.86, 'lower': -1.047, 'upper': 2.094, 'armature': 0.0002},
    'l_ring_finger_dip': {'kp': 0.2285032688178066, 'kv': 0.009917602877441107, 'effort': 0.3, 'velocity': 13.5, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0002},
    'l_pinky_mcp_flex': {'kp': 0.3655325975433942, 'kv': 0.018616960278988272, 'effort': 2.0, 'velocity': 8.203, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0005},
    'l_pinky_mcp_abd': {'kp': 0.41393113081120425, 'kv': 0.018732177029667153, 'effort': 0.2, 'velocity': 8.11, 'lower': -0.698, 'upper': 0.698, 'armature': 0.0002},
    'l_pinky_pip': {'kp': 0.22729367621965954, 'kv': 0.00951005441616486, 'effort': 0.3, 'velocity': 12.86, 'lower': -1.047, 'upper': 2.094, 'armature': 0.0002},
    'l_pinky_dip': {'kp': 0.1964723550341816, 'kv': 0.009017295241756363, 'effort': 0.3, 'velocity': 13.5, 'lower': -1.047, 'upper': 1.57, 'armature': 0.0002},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_payload(asset_dir=C.ASSET_DIR):
    """Reject modified manifests or generated payloads, without relabeling the asset."""
    asset_dir = Path(asset_dir)
    path = asset_dir / "manifest.json"
    if sha256(path) != C.MANIFEST_SHA256:
        raise ValueError("Hand2 diagnostic manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if (manifest["model_pin"] != C.MODEL_PIN or manifest["model"] != "Hand2.2 LEFT"
            or manifest["status"] != "preparatory"
            or manifest["approved_for_retargeting"] or manifest["approved_for_training"]
            or manifest["source_files_sha256"][C.MJCF_SOURCE] != C.MJCF_SHA256):
        raise ValueError("Hand2 preparatory model/source contract mismatch")
    pairs = [(p["body1"], p["body2"]) for p in manifest["vendor_mjcf_exclusions"]["pairs"]]
    if pairs != C.EXPECTED_EXCLUSIONS or len(set(pairs)) != 15:
        raise ValueError("Hand2 source exclusions mismatch")
    for relative, expected in manifest["generated_files_sha256"].items():
        path = asset_dir / relative
        if not path.resolve().is_relative_to(asset_dir.resolve()) or sha256(path) != expected:
            raise ValueError(f"Hand2 generated payload hash mismatch: {relative}")
    return manifest


def derive_parent_weld_pairs(root=None):
    """Derive collision-body pairs from the pinned URDF's fixed-link components.

    Articulated edges join adjacent components. No world component is introduced:
    l_mount is floating, so MuJoCo's world-weld exemption does not apply here.
    An optional XML root permits malformed-graph tests without asset writes.
    """
    if root is None:
        root = ET.parse(C.ROBOT_URDF).getroot()
    links = root.findall("link")
    names = {link.attrib["name"] for link in links}
    if len(names) != len(links) or "l_mount" not in names:
        raise ValueError("Duplicate links or unknown root in Hand2 topology")
    parent = {}
    for joint in root.findall("joint"):
        a, b = (joint.find(k).attrib["link"] for k in ("parent", "child"))
        kind = joint.attrib["type"]
        if a not in names or b not in names or b in parent or kind not in ("fixed", "revolute"):
            raise ValueError("Unknown link, duplicate parent, or unsupported joint in Hand2 topology")
        parent[b] = (a, kind)
    components, active = {}, set()
    def component(name):
        if name in active:
            raise ValueError("Cycle in Hand2 topology")
        if name not in components:
            active.add(name)
            if name not in parent:
                if name != "l_mount":
                    raise ValueError("Unknown root in Hand2 topology")
                components[name] = name
            else:
                ancestor, kind = parent[name]
                ancestor_component = component(ancestor)
                components[name] = ancestor_component if kind == "fixed" else name
            active.remove(name)
        return components[name]
    for name in names:
        component(name)
    if "l_mount" in parent:
        raise ValueError("Hand2 l_mount must be the sole root")
    adjacent = {frozenset((components[a], components[b]))
                for b, (a, kind) in parent.items() if kind != "fixed"}
    collision_names = sorted(link.attrib["name"] for link in links if link.findall("collision"))
    return [(a, b) for a, b in itertools.combinations(collision_names, 2)
            if components[a] == components[b]
            or frozenset((components[a], components[b])) in adjacent]



def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_parameters(parameters=PARAMETERS):
    if tuple(parameters) != JOINT_NAMES or len(parameters) != 20:
        raise ValueError("Expected exactly 20 canonical vendor joints")
    for name, values in parameters.items():
        if set(values) != {"kp", "kv", "effort", "velocity", "lower", "upper", "armature"}:
            raise ValueError(f"Malformed parameters: {name}")
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Nonfinite parameters: {name}")
        if any(values[k] <= 0 for k in ("kp", "kv", "effort", "velocity", "armature")):
            raise ValueError(f"Nonpositive actuator parameter: {name}")
        if not values["lower"] < 0 < values["upper"]:
            raise ValueError(f"Invalid joint limits: {name}")


def validate_asset():
    validate_parameters()
    manifest = validate_payload()
    if sha256(C.ROBOT_URDF) != ASSET_URDF_SHA256:
        raise ValueError("Hand2 asset URDF hash mismatch")
    root = ET.parse(C.ROBOT_URDF).getroot()
    joints = [j for j in root.findall("joint") if j.get("type") != "fixed"]
    if [j.get("name") for j in joints] != list(JOINT_NAMES):
        raise ValueError("Hand2 URDF joint names/order mismatch")
    for joint in joints:
        values, limit = PARAMETERS[joint.get("name")], joint.find("limit")
        for key in ("lower", "upper", "velocity", "effort"):
            if float(limit.get(key)) != values[key]:
                raise ValueError(f"URDF parameter mismatch: {joint.get('name')}/{key}")
    return manifest


def model_layout():
    root = ET.parse(C.ROBOT_URDF).getroot()
    links = [link.get("name") for link in root.findall("link")]
    collisions = [link.get("name") for link in root.findall("link") if link.findall("collision")]
    if len(links) != 33 or len(set(links)) != 33 or len(collisions) != 27:
        raise ValueError("Expected 33 frames and 27 collision links")
    pairs = sorted({tuple(sorted(pair)) for pair in
                    derive_parent_weld_pairs(root) + C.EXPECTED_EXCLUSIONS})
    if any(a not in collisions or b not in collisions or a == b for a, b in pairs):
        raise ValueError("Exclusion is not an exact collision-link pair")
    return links, collisions, pairs


def provenance():
    manifest = validate_asset()
    return {
        "model_pin": C.MODEL_PIN, "mjcf_sha256": C.MJCF_SHA256,
        "source_urdf_sha256": SOURCE_URDF_SHA256, "asset_urdf_sha256": ASSET_URDF_SHA256,
        "manifest_sha256": C.MANIFEST_SHA256,
        "model_payload_sha256": canonical_hash(manifest["generated_files_sha256"]),
        "contract_sha256": sha256(__file__),
        "diagnostic_constants_sha256": sha256(_CONSTANTS_PATH),
        "parameter_sha256": canonical_hash(PARAMETERS),
    }


def safe_path(path):
    if not str(path).strip():
        raise ValueError("Explicit nonempty path required")
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink path rejected: {path}")
    return path


def new_path(path):
    path = safe_path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite {path}")
    if not path.parent.is_dir():
        raise ValueError(f"Parent directory must already exist: {path.parent}")
    return path


def robot_path(path):
    # Deliberately no regex, clone pattern, generic /World root, or object path.
    if not isinstance(path, str) or not re.fullmatch(r"/World/(?:[A-Za-z_][A-Za-z_0-9]*/)*Robot", path):
        raise ValueError("Expected a concrete /World/.../Robot prim path")
    return path


def pose_error(actual, expected):
    """Compare xyz + unit quaternion wxyz, treating q and -q identically."""
    for pose in (actual, expected):
        if len(pose) != 7 or not all(math.isfinite(v) for v in pose):
            raise ValueError("Nonfinite or malformed pose")
        norm = math.sqrt(sum(v*v for v in pose[3:]))
        if abs(norm - 1) > 1e-5:
            raise ValueError("Pose quaternion is not normalized")
    position = math.dist(actual[:3], expected[:3])
    a = [v / math.sqrt(sum(x*x for x in actual[3:])) for v in actual[3:]]
    b = [v / math.sqrt(sum(x*x for x in expected[3:])) for v in expected[3:]]
    chord = min(math.dist(a, b), math.dist(a, [-v for v in b]))
    rotation = 4 * math.asin(min(1.0, chord / 2))
    return position, rotation


def check_pose(actual, expected):
    position, rotation = pose_error(actual, expected)
    if position > POSITION_TOLERANCE_M or rotation > ROTATION_TOLERANCE_RAD:
        raise ValueError(f"FK mismatch: position={position} m, rotation={rotation} rad")
    return position, rotation


def validate_fixture(data):
    """Validate before launching Isaac; no Drake or upstream checkout required."""
    if data.get("schema") != "hand2_left_fk_v1" or data.get("provenance") != provenance():
        raise ValueError("Fixture schema/provenance hash mismatch")
    if data.get("validation_only") is not True or data.get("approved_for_training") is not False:
        raise ValueError("Fixture must remain validation-only")
    builder_hash = data.get("builder_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", builder_hash):
        raise ValueError("Missing builder hash")
    # A deployed runtime need not carry the builder; if present its bytes must match.
    builder = Path(__file__).resolve().parents[6] / "scripts/tactidex_import/assets/build_hand2_left_isaac_fixture.py"
    if builder.is_file() and sha256(builder) != builder_hash:
        raise ValueError("Fixture builder code hash mismatch")
    if data.get("fixture_payload_sha256") != canonical_hash(
            {k: v for k, v in data.items() if k != "fixture_payload_sha256"}):
        raise ValueError("Fixture payload hash mismatch")
    links, _, _ = model_layout()
    if (data.get("joint_names") != list(JOINT_NAMES) or data.get("link_names") != links
            or data.get("mapping") != C.MANO_TO_ROBOT_MAPPING
            or data.get("root_link") != "l_mount" or data.get("wrist_link") != "l_wrist"
            or data.get("fk_position_tolerance_m") != POSITION_TOLERANCE_M
            or data.get("rotation_tolerance_rad") != ROTATION_TOLERANCE_RAD
            or data.get("seed") != 42):
        raise ValueError("Fixture layout/tolerance contract mismatch")
    expected_q_names = [f"l_mount_{s}" for s in ("qw", "qx", "qy", "qz", "x", "y", "z")] + list(JOINT_NAMES)
    if data.get("q_names") != expected_q_names or len(data.get("poses", [])) != 36:
        raise ValueError("Expected 36 poses, native floating 27-q order")
    for index, pose in enumerate(data["poses"]):
        q = pose["q"]
        if len(q) != 27 or not all(math.isfinite(v) for v in q):
            raise ValueError("Malformed q")
        if pose["configuration"] != index // 2 or pose["root_variant"] != index % 2:
            raise ValueError("Unexpected pose order")
        for name, value in zip(JOINT_NAMES, q[7:]):
            if not PARAMETERS[name]["lower"] <= value <= PARAMETERS[name]["upper"]:
                raise ValueError(f"Out-of-limit q: {name}")
        if index < 2 and q[7:] != [0.0] * 20:
            raise ValueError("First configuration must be zero")
        if index % 2 and q[7:] != data["poses"][index-1]["q"][7:]:
            raise ValueError("Root variants must share joints")
        if index % 2 == 0 and q[:7] != [1., 0., 0., 0., 0., 0., 0.]:
            raise ValueError("Identity root missing")
        if index % 2 and q[:7] != data["poses"][1]["q"][:7]:
            raise ValueError("Inconsistent nonidentity root")
        if index % 2 and (q[4:7] != [.17, -.23, .41] or q[1:4] == [0., 0., 0.]):
            raise ValueError("Nonidentity root missing")
        if set(pose["links"]) != set(links) or set(pose["mapping"]) != set(C.MANO_TO_ROBOT_MAPPING):
            raise ValueError("Incomplete FK frames")
        for value in pose["links"].values():
            pose_error(value, value)
        check_pose(pose["root_pose_w"], q[4:7] + q[:4])
        check_pose(pose["root_pose_w"], pose["links"]["l_mount"])
        check_pose(pose["wrist_pose_w"], pose["links"]["l_wrist"])
        for key, name in C.MANO_TO_ROBOT_MAPPING.items():
            check_pose(pose["mapping"][key], pose["links"][name])
    return data
