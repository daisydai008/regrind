"""Isolated MeshTip task IDs; import never authorizes or publishes references."""
import gymnasium as gym

for suffix, config in (("", "Hand2BowlMeshTipSEnvCfg"), ("-Play", "Hand2BowlMeshTipSEnvCfg_PLAY")):
    gym.register(
        id=f"Regrind-WujiHand2-Bowl-MeshTip-S{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hand2_bowl_s_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:Hand2BowlMeshTipSPPORunnerCfg",
        },
    )
