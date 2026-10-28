"""Fresh Scissors PPO; preserve the existing Hand2 architecture and 24-step rollout."""
from isaaclab.utils import configclass
from regrind.tasks.manager_based.dexterous.config.wujihand2.rsl_rl_ppo_cfg import Hand2BowlSPPORunnerCfg


@configclass
class Hand2ScissorsOfficialSPPORunnerCfg(Hand2BowlSPPORunnerCfg):
    seed = 42
    max_iterations = 20000
    experiment_name = "wujihand2_scissors_official_s"
    run_name = ""
    resume = False
    clip_actions = None
