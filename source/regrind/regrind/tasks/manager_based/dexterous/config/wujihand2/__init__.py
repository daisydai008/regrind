"""Independent Hand2 Bowl pilot registrations, discovered by regrind.tasks."""
import gymnasium as gym

for suffix, config in (("", "Hand2BowlSEnvCfg"), ("-Play", "Hand2BowlSEnvCfg_PLAY")):
    gym.register(
        id=f"Regrind-WujiHand2-Bowl-S{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hand2_bowl_s_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:Hand2BowlSPPORunnerCfg",
        },
    )
