from baselines.common import tf_util
from models.ppo.model import Model
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
    with tf.Session(config=tf_config) as session:
        model = Model(session=session, config=config, ob_space=ob_space, ac_space=ac_space)
        tf_util.load_variables(config.load_path, sess=session)
        return make_rollouts(config, env, model)
