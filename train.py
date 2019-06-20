import configargparse
import gym
import models
import numpy as np
import os
import random
import tensorflow as tf
import vizdoomgym


def main():
    config = parse_config()
    env = get_env(config)
    if config.seed is not None:
        set_seed(config.seed)
    algorithm = config.algorithm
    if algorithm == 'baseline_a2c':
        models.run_baseline_a2c(config, env)
    else:
        raise Exception(f'Unknown algorithm: {algorithm}')


def parse_config():
    parser = configargparse.get_arg_parser()
    parser.add('-a', '--algorithm', required=True, help='Algorithm to use. One of: baseline_a2c.')
    parser.add('-e', '--env', required=True, help='Name of Vizdoom. See README for a list of environment names.')
    parser.add('-n', '--name', required=True, help='Name of experiment - used to generate log and output files.')
    parser.add('-t', '--timesteps', required=False, type=int, default=1000000, help='Number of timesteps (default 1 million)')
    parser.add('-bs', '--batch_size', required=False, type=int, default=1000, help='Batch size (default 1000).')
    parser.add('-lr', '--learning_rate', required=False, type=float, default=1e-3, help='Learning rate (default 0.001).')
    parser.add('-s', '--seed', required=False, type=int, default=None, help='Random seed.')
    args = parser.parse_args()
    file_path = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(file_path, 'out')
    log_path = os.path.join(file_path, 'logs')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    args.save_path = os.path.join(out_path, args.name, 'model')
    args.video_path = os.path.join(out_path, args.name)
    args.log_path = os.path.join(log_path, args.name)
    return args


def get_env(config):
    env = gym.make(config.env)
    return env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    main()
