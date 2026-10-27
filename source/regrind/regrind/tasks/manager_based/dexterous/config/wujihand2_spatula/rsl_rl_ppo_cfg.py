"""Fresh formal Spatula training; preserve the existing architecture and std=0.5."""
from isaaclab.utils import configclass
from regrind.tasks.manager_based.dexterous.config.wujihand2.rsl_rl_ppo_cfg import Hand2BowlSPPORunnerCfg


@configclass
class Hand2SpatulaMeshTipSPPORunnerCfg(Hand2BowlSPPORunnerCfg):
    max_iterations = 20000
    experiment_name = "wujihand2_spatula_meshtip_s"
    run_name = ""
    resume = False
