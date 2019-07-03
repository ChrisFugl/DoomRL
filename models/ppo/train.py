from baselines.common import tf_util
from collections import deque
from models.ppo.actor import Actor
from models.ppo.model import Model
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
        model = Model(config, session, env.observation_space, env.action_space)
        actor = Actor(config, env, model)
        info_buffer = deque(maxlen=100)
        update = 1
        timestep = 1
        tstart = time.time()

        while not config.stopping_criterion(timestep):
            observations, actions, advantages, values, neg_log_p, infos = actor.act()
            target = advantages + values
            info_buffer.extend(infos)

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            batches = (observations, actions, target, advantages, values, neg_log_p)

            mb_losses = []
            for epoch in range(config.epochs):
                indices = np.arange(config.batch_size)
                np.random.shuffle(indices)
                for start in range(0, config.batch_size, config.mini_batch_size):
                    end = start + config.mini_batch_size
                    mb_indices = indices[start:end]
                    train_args = (batch[mb_indices] for batch in batches)
                    losses = model.train(*train_args)
                    mb_losses.append(losses)

            # write summaries to tensorboard
            if update % config.log_every == 0 or update == 1:
                seconds = time.time() - tstart
                fps = timestep // seconds
                losses = np.mean(mb_losses, axis=0)
                logger.start_summary()
                logger.add_infos(info_buffer)
                logger.add_value('model/fps', fps)
                logger.add_value('model/total_loss', losses[0])
                logger.add_value('model/policy_loss', losses[1])
                logger.add_value('model/value_loss', losses[2])
                logger.add_value('model/entropy', losses[3])
                logger.log_summary(timestep)

            update += 1
            timestep += config.batch_size

        tf_util.save_variables(config.save_path, sess=session)
