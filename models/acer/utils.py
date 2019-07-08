from functools import partial
import numpy as np
import tensorflow as tf


def batch2rollout(x, n_envs, n_steps):
    """
    :param x: shape (n_envs * n_steps, ...)
    :returns: shape (n_envs, n_steps, ...)
    """
    s = x.shape
    return tf.reshape(x, (n_envs, n_steps, *s[1:]))


def np_rollout2batch(x):
    """
    :param x: shape (n_envs, n_steps, ...)
    :returns: shape (n_envs * n_steps, ...)
    """
    s = x.shape
    return np.reshape(x, (s[0] * s[1], *s[2:]))


def tf_rollout2batch(x):
    """
    :param x: shape (n_envs, n_steps, ...)
    :returns: shape (n_envs * n_steps, ...)
    """
    s = x.shape
    return tf.reshape(x, (s[0] * s[1], *s[2:]))


def rollout2steps(x):
    """
    :param x: shape (n_envs, n_steps, ...)
    :returns: list of length n_steps containing objects of shape (n_envs, ...)
    """
    s = x.shape
    n_steps = s[1]
    steps = tf.split(x, num_or_size_splits=n_steps, axis=1)
    squeeze = partial(tf.squeeze, axis=[1])
    steps = list(map(squeeze, steps))
    return steps


def steps2rollout(x):
    """
    :param x: list of length n_steps containing objects of shape (n_envs, ...)
    :returns: shape (n_envs, n_steps, ...)
    """
    return tf.stack(x, axis=1)
