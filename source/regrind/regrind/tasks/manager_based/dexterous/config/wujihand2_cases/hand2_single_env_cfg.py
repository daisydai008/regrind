"""Native case pilots; validate portable case approval BEFORE parent initialization.

No Bowl approval dependency; official drives, observations and normal terminations
come from Hand2BowlSEnvCfg, not its MeshTip or Hand1 Play variants.
"""
import os
from pathlib import Path

from isaaclab.utils import configclass
from regrind.objects.tactidex_objects import KETTLEBLACK_OBJECT_CFG, SPOON_OBJECT_CFG
from regrind.robots.hand2_case_contract import CASES, bundle_path, validate_bundle
from regrind.tasks.manager_based.dexterous.config.wujihand2.hand2_bowl_s_env_cfg import Hand2BowlSEnvCfg
from regrind.tasks.manager_based.dexterous.mdp.hand2_supported_root import Hand2SupportedRootActionCfg


@configclass
class _Hand2CaseMeshTipSEnvCfg(Hand2BowlSEnvCfg):
    case_name: str = ""

    def __post_init__(self):
        data = bundle_path(self.case_name)
        manifest = validate_bundle(data, self.case_name)
        super().__post_init__()
        case = CASES[self.case_name]
        self.episode_length_s = case["control_frames"] / 30
        factory = {"kettleblack": KETTLEBLACK_OBJECT_CFG, "spoon": SPOON_OBJECT_CFG}[self.case_name]
        # Byte-identical source URDF owns mass (0.5/0.05 kg), not factory mass_hint.
        self.scene.object = factory.replace(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=factory.spawn.replace(
                asset_path=str(data / manifest["object_payload"]["urdf"]),
                usd_dir=str(Path(os.environ["HAND2_USD_CACHE_DIR"]).parent / f"{self.case_name}_meshtip_usd_cache"),
            ),
        )
        self.scene.object.spawn.rigid_props.disable_gravity = False
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        motion = self.commands.motion
        motion.demo_path = str(data / "demo_interp_120fps.h5")
        motion.retargeted_traj_path = str(data / "retargeted_120fps.h5")
        motion.keypoint_paths = {"bottom": str(data / "keypoints.npy")}
        motion.object_pos_lower_bound = tuple(case["lower_bound"])
        motion.object_pos_upper_bound = tuple(case["upper_bound"])
        # Cropped files begin at zero; original source window stays in manifest.
        old = self.actions.root_pose
        params = {key: value for key, value in vars(old).items() if key != "class_type"}
        params.update(body_name="r_mount", root_controller_profile="com_bounded_v3",
                      kp_pos=300.0, kd_pos=30.0, kp_rot=3.0, kd_rot=0.02)
        self.actions.root_pose = Hand2SupportedRootActionCfg(**params)
        self.actions.joint_pos.scale *= 3.0  # Existing V4: (3.2 / 30) * 3 = 0.32 rad.
        # Inherited tabletop=0.012, gravity, no DR/RSI, all normal terms unchanged.


@configclass
class Hand2KettleBlackMeshTipSEnvCfg(_Hand2CaseMeshTipSEnvCfg):
    case_name: str = "kettleblack"


@configclass
class Hand2KettleBlackMeshTipSEnvCfg_PLAY(Hand2KettleBlackMeshTipSEnvCfg):
    """Identical Train/Play physics and normal terminations."""


@configclass
class Hand2KettleBlackMeshTipWrenchV4SEnvCfg(Hand2KettleBlackMeshTipSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.root_pose = self.actions.root_pose.replace(root_controller_profile="com_wrench_bounded_v4")


@configclass
class Hand2KettleBlackMeshTipWrenchV4SEnvCfg_PLAY(Hand2KettleBlackMeshTipWrenchV4SEnvCfg):
    """Identical Train/Play physics and normal terminations."""


@configclass
class Hand2SpoonMeshTipSEnvCfg(_Hand2CaseMeshTipSEnvCfg):
    case_name: str = "spoon"


@configclass
class Hand2SpoonMeshTipSEnvCfg_PLAY(Hand2SpoonMeshTipSEnvCfg):
    """Identical Train/Play physics and normal terminations."""
