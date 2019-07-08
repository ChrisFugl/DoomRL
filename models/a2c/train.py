from baselines.common import tf_util
from collections import deque
from models.a2c.model import Model
from models.a2c.runner import Runner
import os
import random
import tensorflow as tf
import time


def train(config, env, logger):
    tf.reset_default_graph()
    gpu_opts = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        gpu_options=gpu_opts,
    )
    with tf.Session(config=tf_config) as sess:
        model = Model(config, env, session)
        runner = Runner(env, model, config)
        info_buffer = deque(maxlen=100)
        tstart = time.time()
        update = 1
        timestep = 1
        while not config.stopping_criterion(timestep):
            obs, rewards, masks, actions, values, infos = runner.run()
            rewards = rewards / config.reward_scale
            total_loss, policy_loss, value_loss, entropy = model.train(obs, rewards, masks, actions, values)
            info_buffer.extend(infos)

            # write summaries to tensorboard
            if update % config.log_every == 0 or update == 1:
                nseconds = time.time() - tstart
                fps = int(timestep / nseconds)
                logger.summary(timestep, fps, info_buffer, total_loss, policy_loss, value_loss, entropy)

            update += 1
            timestep += config.batch_size

        tf_util.save_variables(config.save_path, sess=sess)


def add_to_summary(summary, scope, name, value):
    summary.value.add(tag=f'{scope}/{name}', simple_value=value)
