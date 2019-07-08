from baselines.common import tf_util
from collections import deque
from models.acer.actor import Actor
from models.acer.model import Model
from models.acer.replay import Replay
from models.acer.utils import np_rollout2batch
import numpy as np
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
    with tf.Session(config=tf_config) as session:
        model = Model(config, env, session)
        actor = Actor(config, env, model)
        replay = Replay(config, env)
        info_buffer = deque(maxlen=100)
        update = 1
        timestep = 1
        tstart = time.time()

        while not config.stopping_criterion(timestep):
            total_loss, policy_loss, value_loss, entropy, infos = _train_on_policy(model, actor, replay)
            info_buffer.extend(infos)
            if replay.can_sample():
                n = np.random.poisson(config.replay_ratio)
                for _ in range(n):
                    _train_off_policy(model, replay)

            # write summaries to tensorboard
            if update % config.log_every == 0 or update == 1:
                seconds = time.time() - tstart
                fps = timestep // seconds
                logger.summary(timestep, fps, info_buffer, total_loss, policy_loss, value_loss, entropy)

            timestep += config.batch_size
            update += 1

        tf_util.save_variables(config.save_path, sess=session)


def _train_on_policy(model, actor, replay):
    observations, actions, rewards, dones, mus, infos = actor.act()
    replay.store(observations, actions, rewards, dones, mus)
    observations, actions, rewards, dones, mus = map(np_rollout2batch, (observations, actions, rewards, dones, mus))
    total_loss, policy_loss, value_loss, entropy = model.train(observations, actions, rewards, dones, mus)
    return total_loss, policy_loss, value_loss, entropy, infos


def _train_off_policy(model, replay):
    observations, actions, rewards, dones, mus = replay.sample()
    model.train(observations, actions, rewards, dones, mus)
