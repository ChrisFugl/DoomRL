import configargparse
from environment import make_train_env
import gym
import models
import numpy as np
import os
import random
import tensorflow as tf


def main():
    config = parse_config()
    env = make_train_env(config)
    if config.seed is not None:
        set_seed(config.seed)
    algorithm = config.algorithm
    if algorithm == 'baseline_a2c':
        from models.baselines.a2c import train
    elif algorithm == 'baseline_ppo2':
        if config.number_of_steps < 100:
            print("WARNING: number of steps is very small")
        from models.baselines.ppo2 import train
    elif algorithm == 'a2c':
        from models.a2c import train
    else:
        raise Exception(f'Unknown algorithm: {algorithm}')
    train(config, env)


def parse_config():
    parser = configargparse.get_arg_parser()
    parser.add(
        '-a', '--algorithm',
        choices=['baseline_a2c', 'a2c', 'baseline_ppo2'],
        required=True,
        help='Algorithm to use. One of: baseline_a2c, a2c, baseline_ppo2.'
    )
    parser.add('-e', '--env', required=True, help='Name of Vizdoom environment. See README for a list of environment names.')
    parser.add('-n', '--name', required=True, help='Name of experiment - used to generate log and output files.')
    parser.add('-t', '--timesteps', required=False, type=int, default=1000000, help='Number of timesteps (default 1 million)')
    parser.add('-lr', '--learning_rate', required=False, type=float, default=1e-3, help='Learning rate (default 0.001).')
    parser.add('-s', '--seed', required=False, type=int, default=None, help='Random seed.')
    parser.add('-d', '--discount_factor', required=False, type=float, default=0.99, help='Discount factor (default 0.99).')
    parser.add('-ns', '--number_of_steps', required=False, type=int, default=5, help='Number of steps (default 5).')
    parser.add('-ne', '--number_of_environments', required=False, type=int, default=8, help='Number of environments in A2C (default 8).')
    parser.add('-m', '--momentum', required=False, type=float, default=0.0, help='Optimizer momentum (default 0).')
    parser.add('-rmspd', '--rmsp_decay', required=False, type=float, default=0.99, help='RMSprop decay (default 0.99).')
    parser.add('-rmspe', '--rmsp_epsilon', required=False, type=float, default=1e-10, help='RMSprop epsilon (default 1e-10).')
    parser.add('-sc', '--skipcount', required=False, type=int, default=1, help='Number of frames to skip (default 1).')
    parser.add('-dr', '--downscale_ratio', required=False, type=int, default=1.0, help='Down scale ratio (default 1 - no downscaling.)')
    parser.add('-g', '--grayscale', required=False, type=get_bool, default=True, help='Use grayscale (default true).')
    parser.add('-fs', '--framestacking', required=False, type=int, default=0, help='Number of stacked frames (default 0 - do not stack).')
    parser.add('-sv', '--save_video', required=False, type=get_bool, default=True, help='Save videos while training.')
    parser.add('-ew', '--entropy_weight', required=False, type=float, default=0.01, help='Weight of entropy (default 0.01).')
    parser.add('-cw', '--critic_weight', required=False, type=float, default=0.5, help='Wegith of critic loss (default 0.5).')
    parser.add('-mgn', '--max_grad_norm', required=False, type=float, default=None, help='Clips gradients at this norm or higher (default no clipping).')
    parser.add(
        '-sm',
        '--sampling_method',
        choices=['noise', 'categorical', 'epsilon', 'max'],
        required=False,
        default='noise',
        help='Method used for sampling (default noise).'
    )
    parser.add('-ep', '--epsilon', required=False, type=float, default=0.1, help='Epsilon in greedy epsilon sampling (default 0.1).')
    args = parser.parse_args()
    args.batch_size = args.number_of_steps * args.number_of_environments
    file_path = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(file_path, 'out')
    log_path = os.path.join(file_path, 'logs')
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    args.save_path = os.path.join(out_path, args.name, 'model')
    args.video_path = os.path.join(out_path, args.name)
    args.log_path = os.path.join(log_path, args.name)
    return args


def get_bool(value):
    if value == "True" or value == "true":
        return True
    elif value == "False" or value == "false":
        return False
    try:
        return bool(int(value))
    except ValueError:
        return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    main()
