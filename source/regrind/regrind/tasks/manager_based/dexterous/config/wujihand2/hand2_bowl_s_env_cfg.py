"""Fixed, native-speed Hand2 Bowl pilot; not a calibrated dynamics baseline."""
import os
from pathlib import Path

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from regrind.objects.tactidex_objects import BOWL_OBJECT_CFG
from regrind.robots.free_hand2_right_hand import contract, make_hand2_right_training_cfg
from regrind.tasks.manager_based.dexterous.dexterous_env_cfg import DexterousEnvCfg
import regrind.tasks.manager_based.dexterous.mdp as mdp
from regrind.tasks.manager_based.dexterous.mdp.commands import MotionCommand

JOINT_NAMES = list(contract.JOINT_NAMES)
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
TIP_BODIES = tuple(f"r_{finger}_tip" for finger in FINGERS)
PAD_BODIES = tuple(f"r_{finger}_tip_sensor_frame" for finger in FINGERS)
TABLE_SENSOR_NAMES = tuple(f"contact_hand2_{finger}_table" for finger in FINGERS)
CONTROLLER_HYPOTHESIS = {
    "status": "uncalibrated_vendor_fingers_and_root_controller_hypothesis",
    "root_frame": "r_mount", "base_action_source": "motion_target",
    "kp_pos": 300.0, "kd_pos": 30.0, "kp_rot": 3.0, "kd_rot": 0.3,
    "finger_parameter_status": contract.PARAMETER_STATUS,
    "fixed_parameters": True, "augmentation": False, "speed_factor": 1.0,
}


class Hand2MotionCommand(MotionCommand):
    @property
    def current_fingertips_pos(self):
        # FrameTransformer sorts rigid-body prims, not the supplied cfg order.
        data = self.tfs["tf_fingertips"].data
        names = list(data.target_frame_names)
        if len(names) != 5 or len(set(names)) != 5 or set(names) != set(TIP_BODIES):
            raise ValueError(f"Expected exactly five named Hand2 tip frames: {names}")
        import torch
        indices = torch.tensor([names.index(name) for name in TIP_BODIES],
                               dtype=torch.long, device=data.target_pos_w.device)
        return data.target_pos_w.index_select(1, indices) - self._env.scene.env_origins.unsqueeze(1)

    @property
    def target_fingertips_pos(self):
        # Source metadata uses MANO insertion order, not numeric MANO order.
        tips = self.motion.retargeted_hand_keypoints[self.time_steps][:, [4, 8, 12, 16, 20], :]
        return self._apply_traj_aug_to_points(tips, self.time_steps)


@configclass
class Hand2BowlSEnvCfg(DexterousEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.seed = 42
        self.scene.num_envs = int(os.environ.get("HAND2_NUM_ENVS", "32"))
        if self.scene.num_envs < 1:
            raise ValueError("HAND2_NUM_ENVS must be positive")
        self.episode_length_s = 497 / 30
        self.sim.gravity = (0.0, 0.0, -9.81)
        data = Path(os.environ["REGRIND_DATA_DIR"]) / "hand2_right/bowl"
        # Caller supplies a never-used cache beneath its run/artifact directory.
        self.scene.robot = make_hand2_right_training_cfg(
            "{ENV_REGEX_NS}/Robot", os.environ["HAND2_USD_CACHE_DIR"])
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.object = BOWL_OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")
        self.scene.object.spawn.rigid_props.disable_gravity = False
        self.scene.table.init_state.pos = (-0.15, 0.0, 0.012 - self.scene.table.spawn.size[2] / 2)

        # No startup/reset randomization, delay buffers, mass/COM/scale/gain DR or pushes.
        for name in vars(self.events):
            setattr(self.events, name, None)
        self.obs_max_lags = {}
        for group in (self.observations.policy, self.observations.critic):
            group.object_joint = None
            for term in vars(group).values():
                if isinstance(term, ObsTerm):
                    if "delay_key" in term.params:
                        term.params["delay_key"] = None
                    if "apply_noise" in term.params:
                        term.params["apply_noise"] = False

        motion = self.commands.motion
        motion.class_type = Hand2MotionCommand
        motion.actuated_joint_names = JOINT_NAMES.copy()
        motion.tf_asset_names = ["tf_fingertips"]
        motion.demo_path = str(data / "demo_interp_120fps.h5")
        motion.retargeted_traj_path = str(data / "retargeted_120fps.h5")
        motion.demo_dt = 1 / 120
        motion.speed_factor = 1.0
        motion.auxiliary_traj_dir = None
        motion.traj_aug_enabled = False
        motion.enable_reset_perturbation = False
        motion.rsi_enabled = False
        motion.adaptive_sampling_enabled = False
        motion.object_joint_axis = None
        motion.keypoint_paths = {"bottom": str(data / "keypoints.npy")}
        motion.keypoint_body_prim_paths = {"bottom": "{ENV_REGEX_NS}/Object/bottom"}
        motion.keypoint_scale = 1.0
        motion.object_pos_lower_bound = (-0.767, -0.099, 0.0)
        motion.object_pos_upper_bound = (-0.076, 0.304, 0.559)

        root = self.actions.root_pose
        root.body_name = "r_mount"
        root.base_action_source = "motion_target"
        root.scale_pos = self.sim.dt * self.decimation
        root.scale_rot = 3.2 * self.sim.dt * self.decimation
        root.kp_pos, root.kd_pos = 300.0, 30.0
        root.kp_rot, root.kd_rot = 3.0, 0.3
        self.actions.joint_pos.joint_names = JOINT_NAMES.copy()
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.base_action_source = "motion_target"
        self.actions.joint_pos.scale = 3.2 * self.sim.dt * self.decimation
        self.rewards.object_keypoints_pos_error_exp.params["std"] = 0.015
        self.rewards.action_l2_exp.weight = 0.5

        # Replace the generic nested-root sensor; tips have no collision geometry.
        self.scene.contact_forces = None
        if self.enable_contact_sensor or self.include_contact_forces_in_critic_obs:
            self.scene.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/r_.*_tip_sensor_frame",
                history_length=3, debug_vis=self.enable_contact_sensor,
                filter_prim_paths_expr=[],
            )
        if self.include_contact_forces_in_critic_obs:
            self.observations.critic.fingertip_contact_forces_w = ObsTerm(
                func=mdp.contact_sensor_net_normal_forces_w_ordered,
                params={"sensor_name": "contact_forces", "body_names_ordered": PAD_BODIES},
            )
        for key, body in zip(TABLE_SENSOR_NAMES, PAD_BODIES):
            setattr(self.scene, key, ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
                history_length=3, track_air_time=False, debug_vis=False, force_threshold=0.1,
            ))
        self.terminations.hit_table.params["sensor_names"] = list(TABLE_SENSOR_NAMES)
        self.terminations.hit_table.params["force_threshold"] = 10.0
        self.tf_source_prim_path = "{ENV_REGEX_NS}/Robot/r_mount"
        self.tf_target_frames = {"fingertips": [FrameTransformerCfg.FrameCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}", offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ) for body in TIP_BODIES]}
        self.scene.setup_frame_transformers(self.tf_source_prim_path, self.tf_target_frames)
        self.viewer.eye = (0.25, -0.75, 0.65)
        self.viewer.lookat = (-0.4, 0.05, 0.18)


@configclass
class Hand2BowlSEnvCfg_PLAY(Hand2BowlSEnvCfg):
    """Identical physics and terminations; no relaxed play-only success criteria."""
