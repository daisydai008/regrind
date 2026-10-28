"""Auto-discovered Scissors-only tasks; no shared or Hand1 registry changes."""
import gymnasium as gym

for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"Regrind-WujiHand2-Scissors-Official-S{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hand2_scissors_env_cfg:Hand2ScissorsOfficialSEnvCfg{cfg_suffix}",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:Hand2ScissorsOfficialSPPORunnerCfg",
        },
    )
