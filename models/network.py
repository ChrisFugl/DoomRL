import numpy as np
import random
import tensorflow as tf


def conv(inputs, nf, ks, strides, name, activ=None, gain=1.0):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=nf,
        kernel_size=ks,
        strides=(strides, strides),
        activation=None,
        kernel_initializer=tf.orthogonal_initializer(gain=gain),
        name=name
    )


def fc(inputs, units, name, activ=None, gain=1.0):
    return tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=None,
        kernel_initializer=tf.orthogonal_initializer(gain),
        name=name
    )


def conv_to_fc(inputs):
    return tf.layers.flatten(inputs)


def sample_categorical(logits):
    distribution = tf.distributions.Categorical(logits=logits)
    return distribution.sample()


def sample_noise(logits):
    noise = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
    return tf.argmax(logits - tf.log(-tf.log(noise)), -1)


def sample_epsilon_greedy(epsilon, nactions, batch_size):
    shape = tf.constant([batch_size], dtype=tf.int32)

    def sample(logits):
        if random.random() < epsilon:
            return tf.random.uniform(shape=shape, minval=0, maxval=nactions, dtype=tf.int32)
        else:
            return tf.argmax(logits, -1)

    return sample


def get_sampler(config, ac_space, batch_size):
    if config.sampling_method == 'noise':
        return sample_noise
    elif config.sampling_method == 'categorical':
        return sample_categorical
    else:
        return sample_epsilon_greedy(config.epsilon, ac_space.n, batch_size)


class FC:
    def __init__(self, sess, scope, ob_space, ac_space, batch_size, config, reuse=False):
        sample = get_sampler(config, ac_space, batch_size)

        #activation function
        activ = tf.nn.relu

        #input shapes
        X = tf.placeholder(tf.float32, [batch_size, ob_space.shape[0]])
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h1 = fc(X, 32, name='fc1', activ=activ, gain=np.sqrt(2))
            h2 = fc(X, 32, name='fc2', activ=activ, gain=np.sqrt(2))
            pi = fc(h2, ac_space.n, name="pi", activ=None)
            vf = fc(h2, 1, name="vf", activ=None)

        # value prediction
        v0 = vf[:, 0]

        # sampe an action given the policy
        a0 = sample(pi)

        def step(ob):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v

        def value(ob):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CNN:
    def __init__(self, sess, scope, ob_space, ac_space, batch_size, config, reuse=False):
        sample = get_sampler(config, ac_space, batch_size)

        #activation function
        activ = tf.nn.relu

        nh, nw, nc = ob_space.shape
        X = tf.placeholder(tf.float32, [batch_size, nh, nw, nc])

        #sclae the images
        scaled_images = tf.cast(X, tf.float32) / 255.

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h1 = conv(scaled_images, nf=32, ks=8, strides=4, name='c1', activ=activ, gain=np.sqrt(2))
            h2 = conv(h1, nf=64, ks=4, strides=2, name='c2', activ=activ, gain=np.sqrt(2))
            h3 = conv(h2, nf=64, ks=3, strides=1, name='c3', activ=activ, gain=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 512, name='fc1', activ=activ, gain=np.sqrt(2))
            pi = fc(h4, ac_space.n, name="pi", activ=None)
            vf = fc(h4, 1, name="vf", activ=None)

        # value prediction
        v0 = vf[:, 0]

        # sampe an action given the policy
        a0 = sample(pi)

        def step(ob):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v

        def value(ob):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
