"""Local pilot budget, preserving the existing single-hand PPO architecture."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from regrind.utils.rl_cfg import RslRlPpoCustomActorCriticCfg


@configclass
class Hand2BowlSPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 100
    experiment_name = "wujihand2_bowl_s"
    run_name = "pilot_uncalibrated"
    policy = RslRlPpoCustomActorCriticCfg(
        init_noise_std=0.5, actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256, 128], critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu", zero_init_actor_last_layer=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.002,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=0.001, schedule="adaptive",
        gamma=0.998, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )
