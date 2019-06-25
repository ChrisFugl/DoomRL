from baselines.a2c.a2c import learn
from baselines.common.tf_util import get_session
import tensorflow as tf


def run(config, env):
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
        # batch size is nsteps * nenv where nenv is number of environment copies simulated in parallel
        nsteps=config.batch_size // config.number_of_environments,
        epsilon=config.rmsp_epsilon
    )
    return model
