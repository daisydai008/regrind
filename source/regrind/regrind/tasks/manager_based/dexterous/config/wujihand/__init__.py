# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Regrind-WujiHand-WokLid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_woklid_env_cfg:WujiHandWokLidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandWokLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-WokLid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_woklid_env_cfg:WujiHandWokLidEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandWokLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-BeakerLid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_beakerlid_env_cfg:WujiHandBeakerLidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandBeakerLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-BeakerLid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_beakerlid_env_cfg:WujiHandBeakerLidEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandBeakerLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-JuicerLip-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_juicerlip_env_cfg:WujiHandJuicerLipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandJuicerLipPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-JuicerLip-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_juicerlip_env_cfg:WujiHandJuicerLipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandJuicerLipPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-ThermosLid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_thermoslid_env_cfg:WujiHandThermosLidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandThermosLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-ThermosLid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_thermoslid_env_cfg:WujiHandThermosLidEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandThermosLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-PotLid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_potlid_env_cfg:WujiHandPotLidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandPotLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-PotLid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_potlid_env_cfg:WujiHandPotLidEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandPotLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-TeacupLid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_teacuplid_env_cfg:WujiHandTeacupLidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandTeacupLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-TeacupLid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_teacuplid_env_cfg:WujiHandTeacupLidEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandTeacupLidPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Eraser-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_eraser_env_cfg:WujiHandEraserEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandEraserPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Eraser-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_eraser_env_cfg:WujiHandEraserEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandEraserPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Spatula-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_spatula_env_cfg:WujiHandSpatulaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandSpatulaPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Spatula-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_spatula_env_cfg:WujiHandSpatulaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandSpatulaPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-GrindHandle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_grindhandle_env_cfg:WujiHandGrindHandleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandGrindHandlePPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-GrindHandle-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_grindhandle_env_cfg:WujiHandGrindHandleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandGrindHandlePPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Scissors-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_scissors_env_cfg:WujiHandScissorsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandScissorsPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Scissors-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_scissors_env_cfg:WujiHandScissorsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandScissorsPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Screwdriver-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_screwdriver_env_cfg:WujiHandScrewdriverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandScrewdriverPPORunnerCfg",
    },
)

gym.register(
    id="Regrind-WujiHand-Screwdriver-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wuji_screwdriver_env_cfg:WujiHandScrewdriverEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WujiHandScrewdriverPPORunnerCfg",
    },
)
