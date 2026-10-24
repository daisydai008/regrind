"""Hand2.2 LEFT diagnostic resources, standalone model constants; no retargeter registration or calibration.

Body origins are used directly, including actual tips (not distal-origin pads).
The native floating l_mount and vendor joint ordering are never rewritten.
"""
from pathlib import Path

ROBOT_NAME = "wujihand2_left_diagnostic"
PARENT_WELD_ROBOT_NAME = "wujihand2_left_diagnostic_parent_weld"
ROBOT_DOF = 20
MODEL_PIN = "829c9aa27cb12c3a49d5b8d63c9869dd8f656702"
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets/hand2_2_left"
ROBOT_URDF = ASSET_DIR / "vendor_drake.urdf"
MANIFEST_SHA256 = "dd6207be4b2808ffed38037217afc83c78e474b12f13a0be6e1cd07355e5a012"
MJCF_SOURCE = "hand2/hand2_2/body/mjcf/left.xml"
MJCF_SHA256 = "9276aa40127671318ffd8b381cfb21f98d0758c960734bfd3067ac8f486cb4d8"
MANO_JOINTS = [str(i) for i in range(21)]
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
MANO_TO_ROBOT_MAPPING = {"0": "l_wrist"}
for _finger, _ids in zip(FINGERS, ((13, 14, 15, 16), (1, 2, 3, 17),
                                      (4, 5, 6, 18), (10, 11, 12, 19), (7, 8, 9, 20))):
    for _id, _suffix in zip(_ids, ("proximal_abd", "middle", "distal", "tip")):
        MANO_TO_ROBOT_MAPPING[str(_id)] = f"l_{_finger}_{_suffix}"
JOINT_NAMES = [f"l_{finger}_{suffix}" for finger in FINGERS
               for suffix in (("cmc_flex", "cmc_abd", "mcp", "ip") if finger == "thumb"
                              else ("mcp_flex", "mcp_abd", "pip", "dip"))]
MCP_LINKS = [f"l_{finger}_proximal" for finger in FINGERS[1:]]
MCP_IDS = [1, 4, 10, 7]
# Exactly the source MJCF's 15 entries; not inferred from adjacency or contacts.
EXPECTED_EXCLUSIONS = [(f"l_{finger}_{suffix}", "l_wrist") for finger in FINGERS
                       for suffix in ("proximal", "proximal_abd")]
EXPECTED_EXCLUSIONS += [(f"l_{finger}_distal", f"l_{finger}_tip_sensor_frame")
                        for finger in ("index_finger", "middle_finger", "pinky", "ring_finger", "thumb")]

