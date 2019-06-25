from baselines.common import tf_util
import numpy as np
import tensorflow as tf
import os
import random
import time
import gym
import vizdoomgym

from .network import CNN, FC


def find_trainable_variables(scope):
    with tf.variable_scope(scope):
        return tf.trainable_variables()


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


'''
class used to initialize the sample_net (sampling) and train_net (training)
'''
class A2C_Agent:
    def __init__(self, sess, ob_space, ac_space, config, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        self.sess = sess
        self.learning_rate = config.learning_rate

        act_ph = tf.placeholder(shape=[None], name="act", dtype=tf.int32)
        adv_ph = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        rew_ph = tf.placeholder(shape=[None], name="rew", dtype=tf.float32)

        if len(ob_space.shape) == 1:
            sample_net = FC(sess, "a2c_agent", ob_space, ac_space, reuse=False)
            train_net = FC(sess, "a2c_agent", ob_space, ac_space, reuse=True)
        else:
            sample_net = CNN(sess, "a2c_agent", ob_space, ac_space, reuse=False)
            train_net = CNN(sess, "a2c_agent", ob_space, ac_space, reuse=True)

        # Actor
        #logprob_actions_ph = self.get_log_prob(train_net.pi, act_ph)
        logprob_actions_ph = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_net.pi, labels=act_ph)
        actor_loss = tf.reduce_mean(-logprob_actions_ph * adv_ph)
        entropy = tf.reduce_mean(self.get_entropy(train_net.pi))

        # Critic
        critic_loss = tf.losses.mean_squared_error(tf.squeeze(train_net.vf), rew_ph)

        # Total
        total_loss = actor_loss - entropy * ent_coef + critic_loss * vf_coef

        # Update operations
        params = tf.trainable_variables("a2c_agent")
        grads = tf.gradients(total_loss, params)
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=config.rmsp_decay, epsilon=config.rmsp_epsilon)
        _train = trainer.apply_gradients(grads)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            feed_dict = {train_net.X: obs, act_ph: actions, adv_ph: advs, rew_ph: rewards}

            if states != []:
                feed_dict[train_net.S] = states
                feed_dict[train_net.M] = masks

            policy_loss, value_loss, policy_entropy, _ = self.sess.run([actor_loss, critic_loss, entropy, _train], feed_dict)

            return policy_loss, value_loss, policy_entropy

        self.train = train
        self.train_net = train_net
        self.sample_net = sample_net
        self.step = sample_net.step
        self.value = sample_net.value
        self.initial_state = []

        tf.global_variables_initializer().run(session=sess)


    def get_log_prob(self, action_logits, act_ph):
        distribution = tf.distributions.Categorical(logits=action_logits)
        return distribution.log_prob(act_ph)

    def get_entropy(self, logits):
        """
        see:
        https://github.com/openai/baselines/blob/master/baselines/common/distributions.py
        """
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


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
        self.states = model.initial_state
        self.dones = [False for _ in range(config.number_of_environments)]

        self.n_total_rewards = 40
        self.rewards_per_env = [[] for _ in range(config.number_of_environments)]
        self.total_rewards = []
        self.total_rewards_idx = 0

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []

        mb_states = self.states

        for n in range(self.nsteps):
            # Given observations, take action and calculate Value (V(s))
            actions, values, states = self.model.step(self.obs)

            # append experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            #take next action, observe rewards
            obs, rewards, dones, _ = self.env.step(actions)

            self.states = states
            self.dones = dones
            self.obs = obs

            #get total rewards
            for n, (done, reward) in enumerate(zip(dones, rewards)):
                self.rewards_per_env[n].append(reward)
                if done:
                    # save to total rewards
                    total_reward = sum(self.rewards_per_env[n])
                    try:
                        self.total_rewards[self.total_rewards_idx] = total_reward
                    except IndexError:
                        self.total_rewards.append(total_reward)
                    self.rewards_per_env[n] = []
                    self.total_rewards_idx = (self.total_rewards_idx + 1) % self.n_total_rewards

            #check if episode is over
            for n, done in enumerate(self.dones):
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

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, self.total_rewards

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]


def train(config, env):
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5
    log_interval = 100

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

        agent = A2C_Agent(
            sess=sess,
            config=config,
            ob_space=ob_space,
            ac_space=ac_space,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
        )

        runner = Runner(env, agent, config)

        #used to calculate fps
        tstart = time.time()

        nupdates = config.timesteps // config.batch_size
        for update in range(1, nupdates + 1):
            obs, states, rewards, masks, actions, values, total_rewards = runner.run()
            policy_loss, value_loss, policy_entropy = agent.train(obs, states, rewards, masks, actions, values)

            nseconds = time.time() - tstart
            fps = int((update * config.batch_size) / nseconds)

            if update % log_interval == 0 or update == 1:
                print('-' * 25)
                print("{:<15}{:>10}".format("updates", update))
                print("{:<15}{:>10}".format("total_timesteps", update * config.batch_size))
                print("{:<15}{:>10}".format("fps", fps))
                print("{:<15}{:>10.4f}".format("policy_entropy", float(policy_entropy)))
                print("{:<15}{:>10.4f}".format("value_loss", float(value_loss)))
                print("{:<15}{:>10.4f}".format("rewards", np.mean(total_rewards)))

        tf_util.save_variables(config.save_path, sess=sess)
