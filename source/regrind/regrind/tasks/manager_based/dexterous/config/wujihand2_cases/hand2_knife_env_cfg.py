"""Knife staged pilot: original five closed force windows, normal inherited terms.

Portable Knife approval is checked before parent initialization. No Bowl receipt,
Play relaxation, reward suppression, counter-based termination or auto-launch.
"""
import os
from pathlib import Path

from isaaclab.utils import configclass
from regrind.objects.tactidex_objects import KNIFE_OBJECT_CFG
from regrind.robots.hand2_knife_contract import WINDOWS, bundle_path, validate_bundle
from regrind.tasks.manager_based.dexterous.config.wujihand2.hand2_bowl_s_env_cfg import Hand2BowlSEnvCfg
from regrind.tasks.manager_based.dexterous.mdp.hand2_supported_root import Hand2SupportedRootActionCfg
from regrind.tasks.manager_based.dexterous.mdp.terminations import hit_table_outside_window


@configclass
class Hand2KnifeMeshTipSEnvCfg(Hand2BowlSEnvCfg):
    def __post_init__(self):
        data = bundle_path()
        manifest = validate_bundle(data)
        super().__post_init__()
        self.scene.num_envs = int(os.environ.get("HAND2_NUM_ENVS", "4096"))
        self.episode_length_s = 429 / 30
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        self.scene.object = KNIFE_OBJECT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=KNIFE_OBJECT_CFG.spawn.replace(
                asset_path=str(data / manifest["object_payload"]["urdf"]),
                usd_dir=str(Path(os.environ["HAND2_USD_CACHE_DIR"]).parent / "knife_meshtip_usd_cache"),
            ),
        )
        self.scene.object.spawn.rigid_props.disable_gravity = False
        self.scene.table.init_state.pos = (-.15, 0., .002 - self.scene.table.spawn.size[2] / 2)
        motion = self.commands.motion
        motion.demo_path = str(data / "demo_interp_120fps.h5")
        motion.retargeted_traj_path = str(data / "retargeted_120fps.h5")
        motion.keypoint_paths = {"bottom": str(data / "keypoints.npy")}
        motion.object_pos_lower_bound = (-.443, -.277, 0.)
        motion.object_pos_upper_bound = (-.070, .226, .395)
        old = self.actions.root_pose
        self.actions.root_pose = Hand2SupportedRootActionCfg(
            asset_name=old.asset_name, body_name="r_mount", base_action_source=old.base_action_source,
            command_name=old.command_name, raw_clip=old.raw_clip,
            scale_pos=old.scale_pos, scale_rot=old.scale_rot,
            root_controller_profile="com_bounded_v3", kp_pos=300., kd_pos=30., kp_rot=3., kd_rot=.02,
        )
        self.actions.joint_pos.scale *= 3.0  # Native 3.2/30 * 3 = 0.32 rad, no drive changes.
        self.terminations.hit_table.func = hit_table_outside_window
        self.terminations.hit_table.params.update(
            command_name="motion", allowed_windows_s=[(a / 30, b / 30) for a, b in WINDOWS],
        )
        if (self.decimation != 4 or self.sim.dt != 1 / 120
                or tuple(self.sim.gravity) != (0., 0., -9.81)
                or tuple(self.scene.table.spawn.size[:2]) != (1.4, 1.)
                or self.terminations.hit_table.params["force_threshold"] != 10.):
            raise ValueError("Inherited Knife physics/force contract changed")


@configclass
class Hand2KnifeMeshTipSEnvCfg_PLAY(Hand2KnifeMeshTipSEnvCfg):
    """Train/Play share physics, observations, rewards and all normal terminations."""
