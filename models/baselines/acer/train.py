from baselines.common.tf_util import get_session
from baselines.acer.acer import learn
import tensorflow as tf


def train(config, env, logger):
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    session_config.gpu_options.allow_growth = True
    get_session(config=session_config)
    model = learn(
        network='cnn',
        env=env,
        nsteps=config.number_of_steps,
        total_timesteps=config.timesteps,
        q_coef=config.critic_weight,
        ent_coef=config.entropy_weight,
        max_grad_norm=config.max_grad_norm,
        lr=config.learning_rate,
        lrschedule='linear',
        rprop_epsilon=config.rmsp_epsilon,
        rprop_alpha=config.rmsp_decay,
        gamma=config.discount_factor,
        log_interval=config.log_every,
        buffer_size=config.buffer_size,
        replay_ratio=config.replay_ratio,
        replay_start=config.replay_start,
        c=config.clipping_factor,
        trust_region=True,
        delta=config.trust_region_delta,
        alpha=config.momentum,
        seed=config.seed
    )

    model.save(config.save_path)
