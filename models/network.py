import numpy as np
import tensorflow as tf

def conv(inputs, nf, ks, strides, name, activ=None,gain=1.0):
    return tf.layers.conv2d(inputs=inputs,
                            filters=nf,
                            kernel_size=ks,
                            strides=(strides, strides),
                            activation=None,
                            kernel_initializer=tf.orthogonal_initializer(gain=gain),
                            name=name)

def fc(inputs, units, name, activ=None, gain=1.0):
    return tf.layers.dense(inputs=inputs,
                            units=units,
                            activation=None,
                            kernel_initializer=tf.orthogonal_initializer(gain),
                            name=name)

def conv_to_fc(inputs):
    return tf.layers.flatten(inputs)

def sample(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)

class CNN_Net():
    def __init__(self, sess, scope, ob_space, ac_space, reuse=False):

        #activation function
        activ = tf.nn.relu

        #input shapes
        print(ob_space.shape)
        if len(ob_space.shape)==1:
            X = tf.placeholder(tf.float32, [None, ob_space.shape[0]])

            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                h1 = fc(X, 32, name = 'fc1', activ=activ, gain=np.sqrt(2))
                h2 = fc(X, 32, name = 'fc2', activ=activ, gain=np.sqrt(2))
                pi = fc(h2, ac_space.n, name ="pi", activ=None)
                vf = fc(h2, 1, name="vf", activ=None)

        else:
            nh, nw, nc = ob_space.shape
            X = tf.placeholder(tf.float32, [None, nh,nw,nc])

            #sclae the images
            scaled_images = tf.cast(X, tf.float32) / 255.

            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                h1 = conv(scaled_images, nf=32, ks=8, strides=4, name='c1', activ=activ, gain=np.sqrt(2))
                h2 = conv(h1, nf=64, ks=4, strides=2, name='c2', activ=activ, gain=np.sqrt(2))
                h3 = conv(h2, nf=64, ks=3, strides=1, name='c3', activ=activ, gain=np.sqrt(2))
                h3 = conv_to_fc(h3)
                h4 = fc(h3, 512, name = 'fc1', activ=activ, gain=np.sqrt(2))
                pi = fc(h4, ac_space.n, name ="pi", activ=None)
                vf = fc(h4, 1, name="vf", activ=None)

        # value prediction
        v0 = vf[:, 0]

        # sampe an action given the policy
        a0 = sample(pi)

        def step(ob):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v, []

        def value(ob):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value