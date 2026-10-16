import os
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from .wuji_base_env_cfg import WujiHandBaseEnvCfg
from regrind.objects.tactidex_objects import SPOON_OBJECT_CFG
from regrind.tasks.manager_based.dexterous.mdp import randomize_rigid_body_material
from regrind.envs.events import randomize_rigid_object_com


@configclass
class WujiHandSpoonSEnvCfg(WujiHandBaseEnvCfg):
    """Single right-hand TactiDex sequence at native speed (528 frames at 30 fps)."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 17.6

        # rigid object: drop the object-joint observation terms (no joint in rigid trajs)
        self.observations.policy.object_joint = None
        self.observations.critic.object_joint = None

        self.scene.object = SPOON_OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

        self.scene.table.init_state.pos = (0.0, 0.0, 0.012 - self.scene.table.spawn.size[2] / 2)

        self.commands.motion.object_pos_lower_bound = (-0.439, -0.153, 0.0)
        self.commands.motion.object_pos_upper_bound = (-0.055, 0.413, 0.41)

        self.commands.motion.keypoint_paths: dict[str, str] = {
            "bottom": f"{os.environ['REGRIND_DATA_DIR']}/keypoints/spoon_bottom_37.npy",
        }
        self.commands.motion.keypoint_body_prim_paths: dict[str, str] = {
            "bottom": "{ENV_REGEX_NS}/Object/bottom",
        }
        self.commands.motion.keypoint_scale: float = 1.0
        # rigid object: joint axis unused (is_articulated_object auto-detected False)
        self.commands.motion.object_joint_axis = None
        self.scene.setup_frame_transformers(
            tf_source_prim_path=self.tf_source_prim_path,
            tf_target_frames=self.tf_target_frames,
        )

        obj = "spoon"
        self.commands.motion.demo_path = f"{os.environ['REGRIND_DATA_DIR']}/tactidex_demo_v2/{obj}/demo_interp_120fps.h5"
        self.commands.motion.demo_dt = 1.0 / 120.0
        self.commands.motion.retargeted_traj_path = f"{os.environ['REGRIND_DATA_DIR']}/retargeted_traj_v2/wujihand/{obj}/retargeted_120fps.h5"
        self.commands.motion.auxiliary_traj_dir = None
        self.commands.motion.speed_factor = 1.0  # native speed (no time warp)
        self.commands.motion.traj_aug_enabled = False
        self.commands.motion.traj_aug_translation_lower_bound = (-0.05, -0.05, 0.0)
        self.commands.motion.traj_aug_translation_upper_bound = (0.05, 0.05, 0.0)
        self.commands.motion.traj_aug_yaw_range = (-0.5235987756, 0.5235987756)
        # tuned for touch->use trimmed demos (contact starts at t=0): blend earlier than scissors' 0.281/0.4375
        self.commands.motion.traj_aug_start = 0.05
        self.commands.motion.traj_aug_use = 0.4

        self.events = self.events.replace(
            object_com=EventTerm(
                func=randomize_rigid_object_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005)},
                },
            ),
            object_physics_material=EventTerm(
                func=randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                    "static_friction_range": (0.7, 1.3),
                    "dynamic_friction_range": (0.7, 1.3),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 250,
                    "make_consistent": True,
                },
            ),
        )


@configclass
class WujiHandSpoonSEnvCfg_PLAY(WujiHandSpoonSEnvCfg):

    enable_contact_sensor: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.events = self.events.replace(
            curriculum_gravity=None,
            curriculum_random_push_object=None,
            curriculum_random_push_robot=None,
        )
        self.commands.motion.rsi_enabled = False
        self.terminations.object_deviation.params["threshold"] = 1.0
        self.terminations.object_rotation_error = None
        self.terminations.hit_table = None
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.viewer.eye = (1., 1., 1.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.commands.motion.debug_vis = True
        self.sync_contact_sensor_sphere_debug_vis()
