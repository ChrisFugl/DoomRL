from baselines.common.tf_util import initialize
from models.acer.utils import batch2rollout, tf_rollout2batch, rollout2steps, steps2rollout
from models.network import CNN
from models.utils import entropy_probits
import tensorflow as tf


class Model:

    def __init__(self, config, env, session):
        self._config = config
        self._env = env
        self._session = session
        self._batch_size = config.batch_size
        self._action_dtype = env.action_space.dtype
        self._n_actions = env.action_space.n
        self._n_envs = config.number_of_environments
        self._n_steps = config.number_of_steps
        self._max_grad_norm = config.max_grad_norm
        self._learning_rate = config.learning_rate
        self._rmsp_decay = config.rmsp_decay
        self._rmsp_epsilon = config.rmsp_epsilon
        self._momentum = config.momentum
        self._gamma = config.discount_factor
        self._c = config.clipping_factor
        self._cw = config.critic_weight
        self._ew = config.entropy_weight
        self._delta = config.trust_region_delta
        self._build_computation_graph()
        initialize()

    def train(self, observations, actions, rewards, dones, mus):
        total_loss, policy_loss, value_loss, entropy, _ = self._session.run(
            [self._total_loss, self._policy_loss, self._value_loss, self._entropy, self._train],
            {
                self._training_model.X: observations,
                self._averaged_model.X: observations,
                self._actions: actions,
                self._rewards: rewards,
                self._dones: dones,
                self._mus: mus,
            }
        )
        return total_loss, policy_loss, value_loss, entropy

    def step(self, observations):
        """
        Samples a batch of actions given a batch of observations.
        """
        return self._session.run(
            [self._actor_model.a, self._actor_f],
            {
                self._actor_model.X: observations,
            }
        )

    def _build_computation_graph(self):
        # placeholders
        self._actions = tf.placeholder(shape=[self._batch_size], name='actions', dtype=self._action_dtype)
        self._rewards = tf.placeholder(shape=[self._batch_size], name='rewards', dtype=tf.float32)
        self._dones = tf.placeholder(shape=[self._batch_size], name='dones', dtype=tf.float32)
        self._mus = tf.placeholder(shape=[self._batch_size, self._n_actions], name='mus', dtype=tf.float32)

        self._one_hot_actions = tf.one_hot(indices=self._actions, depth=self._n_actions)

        # networks
        with tf.variable_scope('acer', reuse=tf.AUTO_REUSE):
            self._actor_model = CNN(self._config, self._env, self._session, self._n_envs)
            self._training_model = CNN(self._config, self._env, self._session, self._batch_size)
        self._params = tf.trainable_variables('acer')
        self._ema = tf.train.ExponentialMovingAverage(self._momentum)
        self._ema_operation = self._ema.apply(self._params)

        with tf.variable_scope('acer', custom_getter=self._get_averaged_parameter, reuse=True):
            self._averaged_model = CNN(self._config, self._env, self._session, self._batch_size)

        # q-values, statistics and importance sampling
        q = self._training_model.q
        self._actor_f = tf.nn.softmax(self._actor_model.pi)
        f = tf.nn.softmax(self._training_model.pi)
        f_a = tf.nn.softmax(self._averaged_model.pi)
        rho = f / (self._mus + 1e-10)
        f_actions = self._select_actions(f)
        q_actions = self._select_actions(q)
        rho_actions = self._select_actions(rho)
        rho_bar = tf.minimum(1.0, rho_actions)
        V = tf.reduce_sum(q * f, axis=-1)

        # make rollouts and steps
        V_r, q_actions_r, rho_bar_r, rewards_r, dones_r = map(self._to_rollout, (V, q_actions, rho_bar, self._rewards, self._dones))
        V_s, q_actions_s, rho_bar_s, rewards_s, dones_s = map(rollout2steps, (V_r, q_actions_r, rho_bar_r, rewards_r, dones_r))

        # losses
        Q_ret = self._retrace(V_s, q_actions_s, rho_bar_s, rewards_s, dones_s)
        truncated_rho_actions = tf.minimum(self._c, rho_actions)
        advantage_ret = Q_ret - V
        advantage_q = q - tf.expand_dims(V, -1)
        log_f = tf.log(f + 1e-10)
        log_f_actions = tf.log(f_actions + 1e-10)
        relu = tf.nn.relu(1.0 - (self._c / (rho + 1e-10)))
        g_f = log_f_actions * tf.stop_gradient(truncated_rho_actions * advantage_ret)
        g_bias_correction = tf.reduce_sum(log_f * tf.stop_gradient(relu * f * advantage_q), axis=1)
        self._entropy = tf.reduce_mean(entropy_probits(f))
        self._value_loss = 0.5 * tf.reduce_mean(tf.square(tf.stop_gradient(Q_ret) - q_actions))
        self._policy_loss = - (tf.reduce_mean(g_f) + tf.reduce_mean(g_bias_correction))
        self._total_loss = self._policy_loss + self._cw * self._value_loss - self._ew * self._entropy

        # compute gradient of KL-divergence term directly as is done in
        # https://github.com/openai/baselines/blob/master/baselines/acer/acer.py
        k = - f_a / (f + 1e-10)

        # gradients
        grads = self._make_gradients(f, k)
        grads_and_var = self._clip_gradients(grads)
        self._optimize(grads_and_var)

    def _get_averaged_parameter(self, getter, *args, **kwargs):
        return self._ema.average(getter(*args, **kwargs))

    def _select_actions(self, x):
        return tf.reduce_sum(x * self._one_hot_actions, axis=1)

    def _to_rollout(self, x):
        return batch2rollout(x, self._n_envs, self._n_steps)

    def _retrace(self, V, q_actions, rho_bar, r, d):
        """
        Computes retrace as specified in algorithm 2 in Wang et al.

        :param V: V-estimates for actions, list of length n_steps containing floats of shape (n_envs)
        :param q_actions: q estimates for actions, list of length n_steps containing floats of shape (n_envs)
        :param rho_bar: truncated important sampling, list of length n_steps containing floats of shape (n_envs)
        :param r: rewards, list of length n_steps containing floats of shape (n_envs)
        :param d: dones, list of length n_steps containing booleans of shape (n_envs)
        :returns: Q_ret, shape (n_envs * n_steps)
        """
        Q_ret = V[-1]
        Q_ret_s = []
        for i in range(self._n_steps - 1, -1, -1):
            Q_ret = r[i] + self._gamma * Q_ret * (1.0 - d[i])
            Q_ret_s.append(Q_ret)
            Q_ret = rho_bar[i] * (Q_ret - q_actions[i]) + V[i]

        # reverse Q_ret_s to get in increasing step order and transform to batch
        Q_ret_s = Q_ret_s[::-1]
        Q_ret_r = steps2rollout(Q_ret_s)
        Q_ret_b = tf_rollout2batch(Q_ret_r)
        return Q_ret_b


    def _make_gradients(self, f, k):
        g = tf.gradients(-(self._policy_loss - self._ew * self._entropy) * self._batch_size, f)[0]
        k_dot_g = tf.reduce_sum(k * g, axis=-1)
        k_norm_squared = tf.reduce_sum(tf.square(k), axis=-1)
        penalty = tf.maximum(0.0, (k_dot_g - self._delta) / (k_norm_squared + 1e-10))
        penalty = tf.expand_dims(penalty, -1) * k
        g = g - penalty
        policy_grads = tf.gradients(f, self._params, - (g / self._batch_size))
        q_grads = tf.gradients(self._cw * self._value_loss, self._params)
        grads = list(map(self._add_gradients, zip(policy_grads, q_grads)))
        return grads

    def _add_gradients(self, grads):
        g1, g2 = grads
        if g1 is None:
            return g2
        elif g2 is None:
            return g1
        else:
            return g1 + g2

    def _clip_gradients(self, grads):
        if self._max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self._max_grad_norm)
        return list(zip(grads, self._params))

    def _optimize(self, grads_and_var):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate, decay=self._rmsp_decay, epsilon=self._rmsp_epsilon)
        grad_op = optimizer.apply_gradients(grads_and_var)
        with tf.control_dependencies([grad_op]):
            self._train = tf.group(self._ema_operation)
