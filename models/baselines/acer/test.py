from baselines.common import tf_util
from baselines.common.policies import build_policy
from baselines.acer.acer import Model
from models.utils import make_rollouts
import tensorflow as tf


def test(config, env):
    ob_space = env.observation_space
    ac_space = env.action_space
    tf.reset_default_graph()
    gpu_opts = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        gpu_options=gpu_opts,
    )
    with tf.Session(config=tf_config) as sess:
        policy = build_policy(env, 'cnn', estimate_q=True)
        model = Model(
            policy=policy,
            ob_space=ob_space,
            ac_space=ac_space,
            nenvs=config.number_of_environments,
            nsteps=config.number_of_steps,
            ent_coef=config.entropy_weight,
            q_coef=config.critic_weight,
            gamma=config.discount_factor,
            max_grad_norm=config.max_grad_norm,
            lr=config.learning_rate,
            rprop_alpha=config.rmsp_decay,
            rprop_epsilon=config.rmsp_epsilon,
            total_timesteps=config.timesteps,
            lrschedule='linear',
            c=config.clipping_factor,
            trust_region=True,
            alpha=config.momentum,
            delta=config.trust_region_delta
        )
        tf_util.load_variables(config.load_path, sess=sess)
        return make_rollouts(config, env, model)
