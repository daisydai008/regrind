"""Full native fixed Spatula reference for authorized formal training.

The external receipt binds reference and training authorization, not calibrated
dynamics. Train/Play retain the rigid Hand2 parent's normal rewards, terminations
and fixed physics; no other case bundle/approval prerequisite.
"""
import os
from pathlib import Path

from isaaclab.utils import configclass
from regrind.objects.tactidex_objects import SPATULA_OBJECT_CFG
from regrind.robots.hand2_spatula_contract import CONTROLS, bundle_path, validate_bundle
from regrind.tasks.manager_based.dexterous.config.wujihand2.hand2_bowl_s_env_cfg import Hand2BowlSEnvCfg
from regrind.tasks.manager_based.dexterous.mdp.hand2_supported_root import Hand2SupportedRootActionCfg


@configclass
class Hand2SpatulaMeshTipSEnvCfg(Hand2BowlSEnvCfg):
    def __post_init__(self):
        data = bundle_path()
        manifest = validate_bundle(data)  # Fail BEFORE parent initialization/spawning.
        super().__post_init__()
        self.episode_length_s = CONTROLS / 30
        # The byte-identical published URDF owns 0.15 kg / three collision hulls.
        self.scene.object = SPATULA_OBJECT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=SPATULA_OBJECT_CFG.spawn.replace(
                asset_path=str(data / manifest["object_payload"]["urdf"]),
                usd_dir=str(Path(os.environ["HAND2_USD_CACHE_DIR"]).parent / "spatula_meshtip_usd_cache"),
            ),
        )
        self.scene.object.spawn.rigid_props.disable_gravity = False
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        motion = self.commands.motion
        motion.demo_path = str(data / "demo_interp_120fps.h5")
        motion.retargeted_traj_path = str(data / "retargeted_120fps.h5")
        motion.keypoint_paths = {"bottom": str(data / "keypoints.npy")}
        motion.object_pos_lower_bound = (-0.417, -0.286, 0.0)
        motion.object_pos_upper_bound = (0.027, 0.198, 0.433)
        old = self.actions.root_pose
        params = {key: value for key, value in vars(old).items() if key != "class_type"}
        params.update(body_name="r_mount", root_controller_profile="com_wrench_bounded_v4",
                      kp_pos=300.0, kd_pos=30.0, kp_rot=3.0, kd_rot=0.02)
        self.actions.root_pose = Hand2SupportedRootActionCfg(**params)
        self.actions.joint_pos.scale = 0.32
        # Existing V4 bounds PD components (50 N / 1 Nm), NOT total support wrench.
        # No object compensation/teleport, DR/RSI/events or term/reward overrides.


@configclass
class Hand2SpatulaMeshTipSEnvCfg_PLAY(Hand2SpatulaMeshTipSEnvCfg):
    """Identical Train/Play physics and normal termination thresholds."""
