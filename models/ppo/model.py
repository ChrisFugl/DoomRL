from baselines.common.tf_util import initialize
from models.network import CNN
from models.utils import categorical_neg_log_p, entropy as get_entropy
import tensorflow as tf


class Model:

    def __init__(self, config, session, ob_space, ac_space):
        self.config = config
        self.session = session
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.actor_model = CNN(session, 'ppo', ob_space, ac_space, config.number_of_environments, config)
        self.training_model = CNN(session, 'ppo', ob_space, ac_space, config.mini_batch_size, config)
        self.c1 = config.critic_weight
        self.c2 = config.entropy_weight
        self.clip = config.clip_epsilon
        self.lr = config.learning_rate
        self.max_grad_norm = config.max_grad_norm
        self.batch_size = config.mini_batch_size
        self._build_computration_graph()
        initialize()

    def train(self, observations, actions, values_target, advantages, values_old, neg_log_p_old):
        variables = [self.total_loss, self.policy_loss, self.value_loss, self.entropy, self._train]
        fd = {
            self.training_model.X: observations,
            self.advantages: advantages,
            self.actions: actions,
            self.values_old: values_old,
            self.values_target: values_target,
            self.neg_log_p_old: neg_log_p_old,
        }
        total_loss, policy_loss, value_loss, entropy, _ = self.session.run(variables, fd)
        return total_loss, policy_loss, value_loss, entropy

    def step(self, observations):
        fd = {self.actor_model.X: observations}
        return self.session.run([self.actor_model.a, self.actor_model.v, self.actor_model.neg_log_p], fd)

    def value(self, observations):
        return self.actor_model.value(observations)

    def _build_computration_graph(self):
        # placeholders
        self.advantages = tf.placeholder(shape=[self.batch_size], dtype=tf.float32, name='advantages')
        self.actions = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name='actions')
        self.values_old = tf.placeholder(shape=[self.batch_size], dtype=tf.float32, name='values_old')
        self.values_target = tf.placeholder(shape=[self.batch_size], dtype=tf.float32, name='values_target')
        self.neg_log_p_old = tf.placeholder(shape=[self.batch_size], dtype=tf.float32, name='neg_log_p_old')

        # compute loss
        pi = self.training_model.pi
        self.neg_log_p = categorical_neg_log_p(pi, self.actions, self.ac_space.n)
        self.entropy = tf.reduce_mean(get_entropy(pi))

        ratio = tf.exp(self.neg_log_p_old - self.neg_log_p)
        policy_loss_unclipped = - ratio * self.advantages
        policy_loss_clipped = - tf.clip_by_value(ratio, 1.0 - self.clip, 1.0 + self.clip) * self.advantages
        self.policy_loss = tf.reduce_mean(tf.maximum(policy_loss_unclipped, policy_loss_clipped))

        # clip value loss to ensure value function does not diverge too much from a potential good distribution
        value_unclipped = self.training_model.v
        value_clipped = self.values_old + tf.clip_by_value(value_unclipped - self.values_old, - self.clip, self.clip)
        value_squared_diff_unclipped = tf.square(value_unclipped - self.values_target)
        value_squared_diff_clipped = tf.square(value_clipped - self.values_target)
        self.value_loss = tf.reduce_mean(tf.maximum(value_squared_diff_unclipped, value_squared_diff_clipped))

        self.total_loss = self.policy_loss + self.c1 * self.value_loss - self.c2 * self.entropy

        # optimize
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-5)
        params = tf.trainable_variables('ppo')

        # clip the gradients (normalize)
        grads_and_var = optimizer.compute_gradients(self.total_loss, params)
        grads, var = zip(*grads_and_var)
        if self.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads_and_var = list(zip(grads, var))

        self._train = optimizer.apply_gradients(grads_and_var)
