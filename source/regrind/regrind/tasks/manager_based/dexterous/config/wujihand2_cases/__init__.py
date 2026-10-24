"""Isolated Hand2 case registrations; importing does not approve or publish data."""
import gymnasium as gym

for case in ("KettleBlack", "Spoon"):
    for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"Regrind-WujiHand2-{case}-MeshTip-S{suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.hand2_single_env_cfg:Hand2{case}MeshTipSEnvCfg{cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:Hand2{case}MeshTipSPPORunnerCfg",
            },
        )

for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"Regrind-WujiHand2-Knife-MeshTip-S{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hand2_knife_env_cfg:Hand2KnifeMeshTipSEnvCfg{cfg_suffix}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:Hand2KnifeMeshTipSPPORunnerCfg",
        },
    )

for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"Regrind-WujiHand2-KettleBlack-MeshTip-WrenchV4-S{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hand2_single_env_cfg:Hand2KettleBlackMeshTipWrenchV4SEnvCfg{cfg_suffix}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:Hand2KettleBlackMeshTipSPPORunnerCfg",
        },
    )
