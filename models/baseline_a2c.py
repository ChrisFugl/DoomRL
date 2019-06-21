from baselines import logger
from baselines.a2c.a2c import learn
from baselines.common.cmd_util import make_vec_env
from baselines.common.tf_util import get_session
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
import os
import tensorflow as tf


def run(config, env):
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'
    logger.configure(config.log_path)
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    session_config.gpu_options.allow_growth = True
    get_session(config=session_config)
    env = make_vec_env(
        config.env,
        config.env,
        1,  # num_env
        config.seed,
        reward_scale=1.0,
        flatten_dict_observations=False
    )
    env = VecVideoRecorder(env, config.video_path, _save_video_when, video_length=200)
    model, _ = learn(
        env=env,
        total_timesteps=config.timesteps,
        network='cnn',
        lr=config.learning_rate,
        gamma=config.discount_factor,
        # batch size is nsteps * nenv where nenv is number of environment copies simulated in parallel
        nsteps=config.batch_size,
    )
    model.save(config.save_path)


def _save_video_when(t):
    return t % 100000 == 0
