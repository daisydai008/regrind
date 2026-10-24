"""500-iteration pilot only by default; no automatic long-run escalation."""
from isaaclab.utils import configclass
from regrind.tasks.manager_based.dexterous.config.wujihand2.rsl_rl_ppo_cfg import Hand2BowlSPPORunnerCfg


@configclass
class Hand2BowlMeshTipSPPORunnerCfg(Hand2BowlSPPORunnerCfg):
    # Inherit seed=42, 24 steps, 5 epochs, 4 minibatches, .001 adaptive,
    # gamma=.998, lambda=.95, and the unchanged single-hand architecture.
    experiment_name = "wujihand2_bowl_meshtip_s"
    run_name = "pilot_uncalibrated"
    max_iterations = 500
    save_interval = 100
