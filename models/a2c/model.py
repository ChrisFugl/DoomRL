from models.network import CNN
from models.utils import categorical_neg_log_p, entropy_logits
import tensorflow as tf


'''
class used to initialize the sample_net (sampling) and train_net (training)
'''
class Model:

    def __init__(self, config, env, sess):
        self.sess = sess
        self.learning_rate = config.learning_rate
        self.ac_space = ac_space

        act_ph = tf.placeholder(shape=[config.batch_size], name='act', dtype=tf.int32)
        adv_ph = tf.placeholder(shape=[config.batch_size], name='adv', dtype=tf.float32)
        rew_ph = tf.placeholder(shape=[config.batch_size], name='rew', dtype=tf.float32)

        with tf.variable_scope('a2c', reuse=tf.AUTO_REUSE):
            sample_net = CNN(config, env, sess, config.number_of_environments)
            train_net = CNN(config, env, sess, config.batch_size)

        # actor
        neglogprob = categorical_neg_log_p(train_net.pi, act_ph, ac_space.n)
        actor_loss = tf.reduce_mean(neglogprob * adv_ph)
        entropy = tf.reduce_mean(entropy_logits(train_net.pi))

        # value
        value_loss = tf.losses.mean_squared_error(tf.squeeze(train_net.vf), tf.squeeze(rew_ph))

        # total
        total_loss = actor_loss - entropy * config.entropy_weight + value_loss * config.critic_weight

        # update operations
        params = tf.trainable_variables('a2c')
        grads = tf.gradients(total_loss, params)
        if config.max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, config.max_grad_norm)
        grads = list(zip(grads, params))

        trainer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=config.rmsp_decay, epsilon=config.rmsp_epsilon)
        _train = trainer.apply_gradients(grads)

        def train(obs, rewards, masks, actions, values):
            advs = rewards - values
            vars = [total_loss, actor_loss, value_loss, entropy, _train]
            feed_dict = {train_net.X: obs, act_ph: actions, adv_ph: advs, rew_ph: rewards}
            _total_loss, policy_loss, _value_loss, policy_entropy, _ = self.sess.run(vars, feed_dict)
            return _total_loss, policy_loss, _value_loss, policy_entropy

        self.train = train
        self.train_net = train_net
        self.sample_net = sample_net
        self.step = sample_net.step
        self.value = sample_net.value

        tf.global_variables_initializer().run(session=sess)
