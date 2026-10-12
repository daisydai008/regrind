"""Rigid TactiDex object assets (single-body, URDF-spawned).

Used by the TactiDex multi-sequence tasks (eraser / spatula / grindhandle).
Each URDF has one link named ``bottom`` (matches the keypoint prim convention).
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from regrind.assets import REGRIND_ASSETS_DIR


def _tactidex_rigid_cfg(name: str, mass_hint: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=f"{REGRIND_ASSETS_DIR}/tactidex/{name}/{name}.urdf",
            # /tmp/IsaacLab on the shared RTX box is owned by another user; keep the
            # converted USD in the caller's own cache (also enables lazy reuse).
            usd_dir=os.path.expanduser(f"~/.cache/regrind_usd/{name}"),
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


ERASER_OBJECT_CFG = _tactidex_rigid_cfg("blackboarderaser", 0.1)
SPATULA_OBJECT_CFG = _tactidex_rigid_cfg("spatula", 0.15)
GRINDHANDLE_OBJECT_CFG = _tactidex_rigid_cfg("grindhandle", 0.2)
WOKLID_OBJECT_CFG = _tactidex_rigid_cfg("woklid", 0.2)
BEAKERLID_OBJECT_CFG = _tactidex_rigid_cfg("beakerlid", 0.1)
JUICERLIP_OBJECT_CFG = _tactidex_rigid_cfg("juicerlip", 0.15)
THERMOSLID_OBJECT_CFG = _tactidex_rigid_cfg("thermoslid", 0.15)
POTLID_OBJECT_CFG = _tactidex_rigid_cfg("potlid", 0.1)
TEACUPLID_OBJECT_CFG = _tactidex_rigid_cfg("teacuplid", 0.1)
