"""Floating Wuji left hand, imported from the calibrated retargeting URDF."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from regrind.assets import REGRIND_ASSETS_DIR

# Canonical retarget/URDF order: finger 1→5, joint 1→4 (20 DOF).
# Isaac imports this asset in joint-major order. Resolve these names at runtime
# with find_joints(..., preserve_order=True) before applying retarget joint targets.
ACTUATED_JOINT_NAMES = [
    f"left_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)
]

# Distal rigid bodies for per-finger contact reporting (thumb=finger1 … pinky=finger5).
WUJI_FINGERTIP_BODY_NAMES: tuple[str, ...] = tuple(f"left_finger{i}_link4" for i in range(1, 6))

RELATIVE_ACTION_SCALE = {
    r"left_finger[1-5]_joint[1-4]": 3.2,  # initial scale; tune after sim tests
}

# Mirror free_wuji_right_hand's actuator gains, with left joint names.
# The right configuration derives these kp/kv values from the upstream MJCF.
_WUJI_LEFT_MJCF_KP_KV = [
    (0.40844710645048043, 0.020882010257063675),
    (0.6858601063643346, 0.030610939996373314),
    (0.2391112196099482, 0.010181475050560962),
    (0.20736128319711747, 0.00909698844675045),
    (0.37352218155860073, 0.01882274330029718),
    (0.45592448909027794, 0.019798167597016643),
    (0.24368366649522863, 0.010477031953162727),
    (0.18026971340925335, 0.008240212147903584),
    (0.3687093483646485, 0.01848622487024593),
    (0.4164253443634641, 0.018032947229953678),
    (0.22218607502059182, 0.009592200014076666),
    (0.19427606072023446, 0.009152994605972402),
    (0.35718151495111794, 0.018376606800780005),
    (0.42977315313086895, 0.01867700966212433),
    (0.24930151196247122, 0.01059512121009555),
    (0.2285032688178066, 0.009917602877441107),
    (0.3655325975433942, 0.018616960278988272),
    (0.41393113081120425, 0.018732177029667153),
    (0.22729367621965954, 0.00951005441616486),
    (0.1964723550341816, 0.009017295241756363),
]


_WUJI_STIFFNESS = {n: t[0] for n, t in zip(ACTUATED_JOINT_NAMES, _WUJI_LEFT_MJCF_KP_KV)}
_WUJI_DAMPING = {n: t[1] for n, t in zip(ACTUATED_JOINT_NAMES, _WUJI_LEFT_MJCF_KP_KV)}

# Preserve the calibrated base frame, fixed fingertip links and MANO anchors.
# The URDF importer places links directly under the asset prim (no USD /root wrapper).
# Runtime joint order is importer-defined: resolve ACTUATED_JOINT_NAMES by name.
FREE_WUJI_LEFT_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{REGRIND_ASSETS_DIR}/mjcf/free_wuji_left_hand_full_simplified_v1.urdf",
        usd_dir=os.path.expanduser("~/.cache/regrind_usd/free_wuji_left_hand"),
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            drive_type="force",
            max_effort=1.0e6,
            max_velocity=1000.0,
        ),  # Need to set this for position control
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[r"left_finger[1-5]_joint[1-4]"],
            stiffness={k: v * 400 for k, v in _WUJI_STIFFNESS.items()},  # *400 for high stiffness since the real motors are not backdrivable
            damping={k: v * 400 for k, v in _WUJI_DAMPING.items()},
            effort_limit_sim=None,
            velocity_limit_sim=3.14,
            armature=None,
            friction=None,
        ),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            f"left_finger{i}_joint{j}": (0.06 if j == 1 else 0.0)
            for i in range(1, 6)
            for j in range(1, 5)
        }
    ),
    soft_joint_pos_limit_factor=1.0,
)
