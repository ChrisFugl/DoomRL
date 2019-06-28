from baselines.common import tf_util
from collections import deque
from models.a2c.model import Model
from models.a2c.runner import Runner
from models.utils import is_not_none, safe_divide, safe_mean
import os
import random
import tensorflow as tf
import time


def train(config, env):
    log_interval = 100
    tb_path = os.path.join(config.log_path, 'tb')
    os.makedirs(tb_path, exist_ok=True)
    writer = tf.summary.FileWriter(tb_path)

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
        model = Model(sess=sess, config=config, ob_space=ob_space, ac_space=ac_space)
        runner = Runner(env, model, config)
        info_buffer = deque(maxlen=100)

        #used to calculate fps
        tstart = time.time()

        nupdates = config.timesteps // config.batch_size
        for update in range(1, nupdates + 1):
            obs, rewards, masks, actions, values, infos = runner.run()
            total_loss, policy_loss, critic_loss, policy_entropy = model.train(obs, rewards, masks, actions, values)
            info_buffer.extend(infos)

            # write summaries to tensorboard
            if update % log_interval == 0 or update == 1:
                timestep = update * config.batch_size
                nseconds = time.time() - tstart
                fps = int((update * config.batch_size) / nseconds)
                ammo_lost = [info['ammo_lost'] for info in info_buffer]
                hits_given = [info['hits_given'] for info in info_buffer]
                accuracy = list(filter(is_not_none, map(safe_divide, zip(hits_given, ammo_lost))))

                summary = tf.Summary()

                add_to_summary(summary, 'model', 'fps', fps)
                add_to_summary(summary, 'model', 'total_loss', total_loss)
                add_to_summary(summary, 'model', 'critic_loss', critic_loss)
                add_to_summary(summary, 'model', 'policy_loss', policy_loss)
                add_to_summary(summary, 'model', 'entropy', policy_entropy)

                add_to_summary(summary, 'episode_means', 'reward', safe_mean([info['reward'] for info in info_buffer]))
                add_to_summary(summary, 'episode_means', 'length', safe_mean([info['length'] for info in info_buffer]))

                add_to_summary(summary, 'game_variables', 'accuracy', safe_mean(accuracy))
                add_to_summary(summary, 'game_variables', 'ammo_gained', safe_mean([info['ammo_gained'] for info in info_buffer]))
                add_to_summary(summary, 'game_variables', 'ammo_lost', safe_mean(ammo_lost))
                add_to_summary(summary, 'game_variables', 'hits_given', safe_mean(hits_given))
                add_to_summary(summary, 'game_variables', 'hits_taken', safe_mean([info['hits_taken'] for info in info_buffer]))
                add_to_summary(summary, 'game_variables', 'health_gained', safe_mean([info['health_gained'] for info in info_buffer]))
                add_to_summary(summary, 'game_variables', 'health_lost', safe_mean([info['health_lost'] for info in info_buffer]))
                add_to_summary(summary, 'game_variables', 'deaths', safe_mean([info['deaths'] for info in info_buffer]))
                add_to_summary(summary, 'game_variables', 'frags', safe_mean([info['frags'] for info in info_buffer]))
                add_to_summary(summary, 'game_variables', 'kills', safe_mean([info['kills'] for info in info_buffer]))

                writer.add_summary(summary, timestep)
                writer.flush()

        tf_util.save_variables(config.save_path, sess=sess)


def add_to_summary(summary, scope, name, value):
    summary.value.add(tag=f'{scope}/{name}', simple_value=value)
