from baselines.a2c.a2c import learn
from baselines.common.tf_util import get_session
import tensorflow as tf


def train(config, env, logger):
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    session_config.gpu_options.allow_growth = True
    get_session(config=session_config)
    model = learn(
        env=env,
        total_timesteps=config.timesteps,
        network='cnn',
        lr=config.learning_rate,
        alpha=config.rmsp_decay,
        gamma=config.discount_factor,
        nsteps=config.number_of_steps,
        epsilon=config.rmsp_epsilon,
        max_grad_norm=config.max_grad_norm,
        ent_coef=config.entropy_weight,
        vf_coef=config.critic_weight,
        log_interval=config.log_every
    )
    model.save(config.save_path)
