"""Separate native pilot experiments; no resume weights or Bowl closing prior."""
from isaaclab.utils import configclass
from regrind.tasks.manager_based.dexterous.config.wujihand2.rsl_rl_ppo_cfg import Hand2BowlSPPORunnerCfg


@configclass
class Hand2KettleBlackMeshTipSPPORunnerCfg(Hand2BowlSPPORunnerCfg):
    experiment_name = "wujihand2_kettleblack_meshtip_s"
    # Base: seed 42, max_iterations 500, save_interval 100, zero actor, std 0.5.


@configclass
class Hand2SpoonMeshTipSPPORunnerCfg(Hand2BowlSPPORunnerCfg):
    experiment_name = "wujihand2_spoon_meshtip_s"
