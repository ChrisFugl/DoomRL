from models.network import CNN, FC
import tensorflow as tf


'''
class used to initialize the sample_net (sampling) and train_net (training)
'''
class Model:
    def __init__(self, sess, ob_space, ac_space, config):
        self.sess = sess
        self.learning_rate = config.learning_rate
        self.ac_space = ac_space

        act_ph = tf.placeholder(shape=[config.batch_size], name='act', dtype=tf.int32)
        adv_ph = tf.placeholder(shape=[config.batch_size], name='adv', dtype=tf.float32)
        rew_ph = tf.placeholder(shape=[config.batch_size], name='rew', dtype=tf.float32)

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
        critic_loss = tf.losses.mean_squared_error(tf.squeeze(train_net.vf), tf.squeeze(rew_ph))

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
