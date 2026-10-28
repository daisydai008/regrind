"""Fixed official articulated Scissors, Hand2 RIGHT; uncalibrated dynamics.

Direct DexterousEnvCfg inheritance: no rigid-case bundle/Hand1 initialization.
Native 641-row H5 remains intact; the existing loader resamples to 270 controls.
"""
import os

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from regrind.objects.scissors import REAL_SCISSORS_OBJECT_CFG
from regrind.robots.free_hand2_right_hand import contract, make_hand2_right_training_cfg
from regrind.robots.hand2_scissors_contract import CONTROLS, SPEED, bundle_path, validate_bundle
from regrind.tasks.manager_based.dexterous.dexterous_env_cfg import DexterousEnvCfg
import regrind.tasks.manager_based.dexterous.mdp as mdp
from regrind.tasks.manager_based.dexterous.mdp.commands import MotionCommand
from regrind.tasks.manager_based.dexterous.mdp.hand2_supported_root import Hand2SupportedRootActionCfg

JOINT_NAMES = list(contract.JOINT_NAMES)
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
TIP_BODIES = tuple(f"r_{finger}_tip" for finger in FINGERS)
PAD_BODIES = tuple(f"r_{finger}_tip_sensor_frame" for finger in FINGERS)
TABLE_SENSOR_NAMES = tuple(f"contact_hand2_{finger}_table" for finger in FINGERS)


class Hand2ScissorsMotionCommand(MotionCommand):
    @property
    def current_fingertips_pos(self):
        # FrameTransformer may reorder rigid prims; never assume cfg order.
        import torch
        data = self.tfs["tf_fingertips"].data
        names = list(data.target_frame_names)
        if len(names) != 5 or len(set(names)) != 5 or set(names) != set(TIP_BODIES):
            raise ValueError(f"Expected five named Hand2 tip frames: {names}")
        indices = torch.tensor([names.index(name) for name in TIP_BODIES], dtype=torch.long, device=data.target_pos_w.device)
        return data.target_pos_w.index_select(1, indices) - self._env.scene.env_origins.unsqueeze(1)

    @property
    def target_fingertips_pos(self):
        tips = self.motion.retargeted_hand_keypoints[self.time_steps][:, [4, 8, 12, 16, 20], :]
        return self._apply_traj_aug_to_points(tips, self.time_steps)


@configclass
class Hand2ScissorsOfficialSEnvCfg(DexterousEnvCfg):
    def __post_init__(self):
        data = bundle_path()
        validate_bundle(data)  # Before parent configuration, conversion or spawning.
        super().__post_init__()
        self.seed = 42
        self.scene.num_envs = int(os.environ.get("HAND2_NUM_ENVS", "32"))
        if self.scene.num_envs < 1:
            raise ValueError("HAND2_NUM_ENVS must be positive")
        self.episode_length_s = CONTROLS / 30
        self.sim.gravity = (0., 0., -9.81)
        self.scene.robot = make_hand2_right_training_cfg("{ENV_REGEX_NS}/Robot", os.environ["HAND2_USD_CACHE_DIR"])
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.object = REAL_SCISSORS_OBJECT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=REAL_SCISSORS_OBJECT_CFG.spawn.replace(usd_path=str(data / "object/scissors_flattened.usd")),
        )
        self.scene.object.spawn.rigid_props.disable_gravity = False
        # REAL_SCISSORS_OBJECT_CFG owns joint friction .2; not a surface material.
        self.scene.table.init_state.pos = (0., 0., .952 - self.scene.table.spawn.size[2] / 2)
        for name in vars(self.events):
            setattr(self.events, name, None)
        for name in vars(self.curriculum):
            setattr(self.curriculum, name, None)
        self.obs_max_lags = {}
        for group in (self.observations.policy, self.observations.critic):
            group.enable_corruption = False
            # Keep articulated object_joint in BOTH observation groups.
            for term in vars(group).values():
                if isinstance(term, ObsTerm):
                    term.noise = None
                    if "delay_key" in term.params:
                        term.params["delay_key"] = None
                    if "apply_noise" in term.params:
                        term.params["apply_noise"] = False
        motion = self.commands.motion
        motion.class_type = Hand2ScissorsMotionCommand
        motion.actuated_joint_names = JOINT_NAMES.copy()
        motion.tf_asset_names = ["tf_fingertips"]
        motion.demo_path = str(data / "demo_interp_120fps.h5")
        motion.retargeted_traj_path = str(data / "retargeted_120fps.h5")
        motion.demo_dt = 1 / 120
        motion.speed_factor = SPEED
        motion.auxiliary_traj_dir = None
        motion.traj_aug_enabled = False
        motion.enable_reset_perturbation = False
        motion.rsi_enabled = False
        motion.adaptive_sampling_enabled = False
        motion.object_joint_axis = (0., 0., -1.)
        motion.keypoint_paths = {part: str(data / f"keypoints_{part}.npy") for part in ("bottom", "top")}
        motion.keypoint_body_prim_paths = {part: f"{{ENV_REGEX_NS}}/Object/bottom/{part}" for part in ("bottom", "top")}
        motion.keypoint_scale = 1.
        motion.object_pos_lower_bound = (-.25, -.25, .95)
        motion.object_pos_upper_bound = (.25, .25, 1.5)
        old = self.actions.root_pose
        self.actions.root_pose = Hand2SupportedRootActionCfg(
            asset_name="robot", body_name="r_mount", base_action_source="motion_target",
            command_name=old.command_name, raw_clip=old.raw_clip,
            scale_pos=self.sim.dt*self.decimation, scale_rot=3.2*self.sim.dt*self.decimation,
            root_controller_profile="com_wrench_bounded_v4", kp_pos=300., kd_pos=30., kp_rot=3., kd_rot=.02,
        )
        self.actions.joint_pos.joint_names = JOINT_NAMES.copy()
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.base_action_source = "motion_target"
        self.actions.joint_pos.scale = .32
        self.rewards.object_keypoints_pos_error_exp.params["std"] = .015
        self.rewards.action_l2_exp.weight = .5
        self.scene.contact_forces = None
        if self.enable_contact_sensor or self.include_contact_forces_in_critic_obs:
            self.scene.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/r_.*_tip_sensor_frame", history_length=3,
                debug_vis=self.enable_contact_sensor, filter_prim_paths_expr=[],
            )
        if self.include_contact_forces_in_critic_obs:
            self.observations.critic.fingertip_contact_forces_w = ObsTerm(
                func=mdp.contact_sensor_net_normal_forces_w_ordered,
                params={"sensor_name": "contact_forces", "body_names_ordered": PAD_BODIES},
            )
        for name, body in zip(TABLE_SENSOR_NAMES, PAD_BODIES):
            setattr(self.scene, name, ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}", filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
                history_length=3, track_air_time=False, debug_vis=False, force_threshold=.1,
            ))
        self.terminations.hit_table.params.update(sensor_names=list(TABLE_SENSOR_NAMES), force_threshold=10.)
        self.tf_source_prim_path = "{ENV_REGEX_NS}/Robot/r_mount"
        self.tf_target_frames = {"fingertips": [FrameTransformerCfg.FrameCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}", offset=OffsetCfg(pos=(0., 0., 0.)),
        ) for body in TIP_BODIES]}
        # Install transformers only after both concrete articulations are chosen.
        self.scene.setup_frame_transformers(self.tf_source_prim_path, self.tf_target_frames)
        self.viewer.eye = (1., 1., 1.5)
        self.viewer.lookat = (0., 0., 1.)


@configclass
class Hand2ScissorsOfficialSEnvCfg_PLAY(Hand2ScissorsOfficialSEnvCfg):
    """Identical physics, rewards and all six normal termination thresholds."""
