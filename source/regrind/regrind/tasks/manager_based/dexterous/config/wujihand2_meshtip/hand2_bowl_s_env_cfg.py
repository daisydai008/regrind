"""Bowl-only MeshTip pilot; external reference approval required, no calibration claim.

Run hand2_bowl_meshtip_contract.py preflight BEFORE training AppLauncher. Registry
import is intentionally side-effect free; this config revalidates before super.
"""
from isaaclab.utils import configclass
from regrind.robots.hand2_bowl_meshtip_contract import bundle_path, validate_bundle
from regrind.tasks.manager_based.dexterous.config.wujihand2.hand2_bowl_s_env_cfg import Hand2BowlSEnvCfg
from regrind.tasks.manager_based.dexterous.mdp.hand2_supported_root import Hand2SupportedRootActionCfg


@configclass
class Hand2BowlMeshTipSEnvCfg(Hand2BowlSEnvCfg):
    def __post_init__(self):
        data = bundle_path()
        validate_bundle(data)  # Hard rejection before parent factory/cache checks.
        super().__post_init__()
        # Keep canonical Bowl ten-hull asset, table, bounds, observations, rewards,
        # all normal terminations and official vendor 20-finger drive parameters.
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        motion = self.commands.motion
        motion.demo_path = str(data / "demo_interp_120fps.h5")
        motion.retargeted_traj_path = str(data / "retargeted_120fps.h5")
        motion.keypoint_paths = {"bottom": str(data / "keypoints.npy")}
        old = self.actions.root_pose
        self.actions.root_pose = Hand2SupportedRootActionCfg(
            asset_name=old.asset_name, body_name="r_mount", base_action_source=old.base_action_source,
            command_name=old.command_name, raw_clip=old.raw_clip,
            scale_pos=old.scale_pos, scale_rot=old.scale_rot,
        )
        # Inherited: seed 42, 497/30 s, native speed, no augmentation/RSI/DR.
        # Root stays 6D, fingers stay 20D; no policy/observation layout change.


@configclass
class Hand2BowlMeshTipSEnvCfg_PLAY(Hand2BowlMeshTipSEnvCfg):
    """Identical physics, guards and normal terminations; no Play-only relaxation."""


@configclass
class Hand2BowlMeshTipWristV2SEnvCfg(Hand2BowlMeshTipSEnvCfg):
    """Opt-in nonofficial virtual-wrist hypothesis, not hardware calibration."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.root_pose = self.actions.root_pose.replace(
            root_controller_profile="com_supported_v2",
            kp_pos=300.0, kd_pos=30.0, kp_rot=3.0, kd_rot=0.02,
        )


@configclass
class Hand2BowlMeshTipWristV2SEnvCfg_PLAY(Hand2BowlMeshTipWristV2SEnvCfg):
    """Same v2 physics and guards; no Play-only relaxation."""


@configclass
class Hand2BowlMeshTipWristV3SEnvCfg(Hand2BowlMeshTipWristV2SEnvCfg):
    """SIM virtual wrist: 1 Nm PD saturation, not a combined root torque cap."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.root_pose = self.actions.root_pose.replace(root_controller_profile="com_bounded_v3")


@configclass
class Hand2BowlMeshTipWristV3SEnvCfg_PLAY(Hand2BowlMeshTipWristV3SEnvCfg):
    """Same v3 physics and guards; no Play-only relaxation."""
