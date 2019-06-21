from baselines.common.atari_wrappers import FrameStack
from baselines.common.retro_wrappers import Downsample, Rgb2gray
import configargparse
import gym
import models
import numpy as np
import os
import random
import tensorflow as tf
import vizdoomgym
from vizdoomgym.envs.vizdoomenv import VizdoomEnv


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
    parser.add('-bs', '--batch_size', required=False, type=int, default=256, help='Batch size (default 1000).')
    parser.add('-lr', '--learning_rate', required=False, type=float, default=1e-3, help='Learning rate (default 0.001).')
    parser.add('-s', '--seed', required=False, type=int, default=None, help='Random seed.')
    parser.add('-d', '--discount_factor', required=False, type=float, default=0.99, help='Discount factor (default 0.99).')
    parser.add('-ne', '--number_of_environments', required=False, type=int, default=1, help='Number of environments in A2C (default 1).')
    parser.add('-m', '--momentum', required=False, type=float, default=0.0, help='Optimizer momentum (default 0).')
    parser.add('-rmspd', '--rmsp_decay', required=False, type=float, default=0.99, help='RMSprop decay (default 0.99).')
    parser.add('-rmspe', '--rmsp_epsilon', required=False, type=float, default=1e-10, help='RMSprop epsilon (default 1e-10).')
    parser.add('-sc', '--skipcount', required=False, type=int, default=0, help='Number of frames to skip (default 0).')
    parser.add('-dr', '--downscale_ratio', required=False, type=int, default=1.0, help='Down scale ratio (default 1 - no downscaling.)')
    parser.add('-g', '--grayscale', required=False, type=bool, default=True, help='Use grayscale (default true).')
    parser.add('-fs', '--framestacking', required=False, type=int, default=0, help='Number of stacked frames (default 0 - do not stack).')
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
    if isinstance(env, VizdoomEnv):
        print('SETTING SKIPCOUNT')
        env.set_skipcount(config.skipcount)
    if config.downscale_ratio != 1:
        env = Downsample(env, config.downscale_ratio)
    if config.grayscale:
        env = Rgb2gray(env)
    if config.framestacking != 0:
        env = FrameStack(env, config.framestacking)
    return env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    main()
