from baselines.common.policies import build_policy
from baselines.ppo2.model import Model
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
        nenvs = env.num_envs
        nbatch = nenvs * config.number_of_steps
        nbatch_train = nbatch // 4
        policy = build_policy(env, 'cnn')
        model = Model(
            policy=policy,
            ob_space=ob_space,
            ac_space=ac_space,
            nbatch_act=nenvs,
            nbatch_train=nbatch_train,
            nsteps=config.number_of_steps,
            ent_coef=config.entropy_weight,
            vf_coef=config.critic_weight,
            max_grad_norm=config.max_grad_norm,
            comm=None,
            mpi_rank_weight=1
        )
        model.load(config.load_path)
        return make_rollouts(config, env, model)
