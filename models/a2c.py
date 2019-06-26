from baselines import logger
from baselines.common import tf_util
from collections import deque
import numpy as np
import os
import random
import tensorflow as tf
import time

from .network import CNN, FC


'''
class used to initialize the sample_net (sampling) and train_net (training)
'''
class A2C:
    def __init__(self, sess, ob_space, ac_space, config):
        self.sess = sess
        self.learning_rate = config.learning_rate
        self.ac_space = ac_space

        act_ph = tf.placeholder(shape=[None], name='act', dtype=tf.int32)
        adv_ph = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)
        rew_ph = tf.placeholder(shape=[None], name='rew', dtype=tf.float32)

        if len(ob_space.shape) == 1:
            sample_net = FC(sess, 'a2c_agent', ob_space, ac_space, config.number_of_environments, config, reuse=False)
            train_net = FC(sess, 'a2c_agent', ob_space, ac_space, config.batch_size, config, reuse=True)
        else:
            sample_net = CNN(sess, 'a2c_agent', ob_space, ac_space, config.number_of_environments, config, reuse=False)
            train_net = CNN(sess, 'a2c_agent', ob_space, ac_space, config.batch_size, config, reuse=True)

        # Actor
        neglogprob = self.get_neglog_prob(train_net.pi, act_ph)
        actor_loss = tf.reduce_mean(neglogprob * adv_ph)
        entropy = tf.reduce_mean(self.get_entropy(train_net.pi))

        # Critic
        critic_loss = tf.losses.mean_squared_error(tf.squeeze(train_net.vf), rew_ph)

        # Total
        total_loss = actor_loss - entropy * config.entropy_weight + critic_loss * config.critic_weight

        # Update operations
        params = tf.trainable_variables('a2c_agent')
        grads = tf.gradients(total_loss, params)
        if config.max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, config.max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=config.rmsp_decay, epsilon=config.rmsp_epsilon)
        _train = trainer.apply_gradients(grads)

        def train(obs, rewards, masks, actions, values):
            advs = rewards - values
            vars = [total_loss, actor_loss, critic_loss, entropy, _train]
            feed_dict = {train_net.X: obs, act_ph: actions, adv_ph: advs, rew_ph: rewards}
            _total_loss, policy_loss, _critic_loss, policy_entropy, _ = self.sess.run(vars, feed_dict)
            return _total_loss, policy_loss, _critic_loss, policy_entropy

        self.train = train
        self.train_net = train_net
        self.sample_net = sample_net
        self.step = sample_net.step
        self.value = sample_net.value

        tf.global_variables_initializer().run(session=sess)


    def get_neglog_prob(self, logits, actions):
        labels = tf.one_hot(actions, self.ac_space.n)
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

    def get_entropy(self, logits):
        '''
        see:
        https://github.com/openai/baselines/blob/master/baselines/common/distributions.py
        '''
        a0 = logits - tf.reduce_max(logits, -1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, -1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), -1)


'''
class used to generates a batch of experiences
'''
class Runner:
    def __init__(self, env, model, config):

        self.env = env
        self.model = model

        if len(env.observation_space.shape) == 1:
            self.batch_ob_shape = (config.batch_size, env.observation_space.shape[0])
        else:
            nh, nw, nc = env.observation_space.shape
            self.batch_ob_shape = (config.batch_size, nh, nw, nc)

        self.obs = self.env.reset()
        self.gamma = config.discount_factor
        self.nsteps = config.number_of_steps
        self.dones = [False for _ in range(config.number_of_environments)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_infos = [], [], [], [], [], []

        for n in range(self.nsteps):
            # Given observations, take action and calculate Value (V(s))
            actions, values = self.model.step(self.obs)

            # append experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            #take next action, observe rewards
            obs, rewards, dones, infos = self.env.step(actions)

            for info in infos:
                episode = info.get('episode')
                if episode:
                    mb_infos.append({
                        'reward': episode['r'],
                        'length': episode['l'],
                        'ammo_gained': info.get('ammo_gained'),
                        'ammo_lost': info.get('ammo_lost'),
                        'health_gained': info.get('health_gained'),
                        'health_lost': info.get('health_lost'),
                        'deaths': info.get('deaths'),
                        'frags': info.get('frags'),
                        'kills': info.get('kills'),
                        'hits_given': info.get('hits_given'),
                        'hits_taken': info.get('hits_taken'),
                    })

            self.dones = dones
            self.obs = obs

            #check if episode is over
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0

            mb_rewards.append(rewards)

        mb_dones.append(self.dones)

        # convert batch of steps in different environments to batch of rollouts
        # from shape (steps, envs) to (envs, steps)

        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        #get last values
        last_values = self.model.value(self.obs).tolist()

        # discount
        if self.gamma > 0:
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = self.discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = self.discount_with_dones(rewards, dones, self.gamma)
                mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        return mb_obs, mb_rewards, mb_masks, mb_actions, mb_values, mb_infos

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]


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
        agent = A2C(sess=sess, config=config, ob_space=ob_space, ac_space=ac_space)
        runner = Runner(env, agent, config)
        info_buffer = deque(maxlen=100)

        #used to calculate fps
        tstart = time.time()

        nupdates = config.timesteps // config.batch_size
        for update in range(1, nupdates + 1):
            obs, rewards, masks, actions, values, infos = runner.run()
            total_loss, policy_loss, critic_loss, policy_entropy = agent.train(obs, rewards, masks, actions, values)
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


def is_not_none(x):
    return x is not None


def safe_divide(values):
    a, b = values
    if b == 0:
        return None
    return a / b


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def add_to_summary(summary, scope, name, value):
    summary.value.add(tag=f'{scope}/{name}', simple_value=value)
