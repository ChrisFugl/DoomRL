from baselines.a2c.a2c import learn, Model
from baselines.common.policies import build_policy
from baselines.common.tf_util import get_session
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
        config.batch_size = 2
        config.number_of_steps = 2
        policy = build_policy(env, 'cnn')
        model = Model(
            policy=policy,
            env=env,
            nsteps=config.number_of_steps,
            ent_coef=config.entropy_weight,
            vf_coef=config.critic_weight,
            max_grad_norm=config.max_grad_norm,
            lr=config.learning_rate,
            alpha=config.rmsp_decay,
            epsilon=config.discount_factor,
            total_timesteps=config.timesteps,
            lrschedule='linear'
        )
        model.load(config.load_path)
        return make_rollouts(config, env, model)


def train(config, env):
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
        nsteps=config.number_of_steps,
        epsilon=config.rmsp_epsilon,
        max_grad_norm=config.max_grad_norm,
        ent_coef=config.entropy_weight,
        vf_coef=config.critic_weight
    )
    model.save(config.save_path)
