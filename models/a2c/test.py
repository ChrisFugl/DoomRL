from baselines.common import tf_util
from models.a2c.model import Model
from models.utils import make_rollouts
import tensorflow as tf


def test(config, env):
    tf.reset_default_graph()
    gpu_opts = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        gpu_options=gpu_opts,
    )
    with tf.Session(config=tf_config) as sess:
        model = Model(config, env, sess)
        tf_util.load_variables(config.load_path, sess=sess)
        return make_rollouts(config, env, model)
