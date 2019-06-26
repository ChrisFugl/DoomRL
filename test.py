import configargparse
from environment import make_test_env
import gym
import models
import numpy as np
import os
import random
import tensorflow as tf


def main():
    config = parse_config()
    env = make_test_env(config)
    if config.seed is not None:
        set_seed(config.seed)
    algorithm = config.algorithm
    if algorithm == 'baseline_a2c':
        rollouts = models.baseline_a2c.test(config, env)
    elif algorithm == 'baseline_ppo2':
        rollouts = models.baseline_ppo2.test(config, env)
    elif algorithm == 'a2c':
        rollouts = models.a2c.test(config, env)
    else:
        raise Exception(f'Unknown algorithm: {algorithm}')
    save_path = os.path.join(config.out, 'rollouts.csv')
    rollouts.to_csv(save_path, index=None)


def parse_config():
    parser = configargparse.get_arg_parser()
    parser.add(
        '-a', '--algorithm',
        choices=['baseline_a2c', 'a2c', 'baseline_ppo2'],
        required=True,
        help='Algorithm to use. One of: baseline_a2c, a2c, baseline_ppo2.'
    )
    parser.add('-e', '--env', required=True, help='Name of Vizdoom environment. See README for a list of environment names.')
    parser.add('-n', '--name', required=True, help='Name of the model (as set during training).')
    parser.add('-o', '--out', required=True, help='Save path of experiments.')
    parser.add('-sv', '--save_video', required=False, type=get_bool, default=False, help='Save videos of rollouts (default false).')
    parser.add('-s', '--seed', required=False, type=int, default=None, help='Random seed.')
    parser.add('-r', '--rollouts', required=False, type=int, default=100, help='Rollouts per experiment (default 100).')
    parser.add('-dr', '--downscale_ratio', required=False, type=int, default=1.0, help='Down scale ratio (default 1 - no downscaling.)')
    parser.add('-g', '--grayscale', required=False, type=get_bool, default=True, help='Use grayscale (default true).')
    parser.add('-fs', '--framestacking', required=False, type=int, default=0, help='Number of stacked frames (default 0 - do not stack).')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    file_path = os.path.dirname(os.path.realpath(__file__))
    load_path = os.path.join(file_path, 'out')
    args.load_path = os.path.join(load_path, args.name, 'model')
    args.video_path = os.path.join(args.out, 'videos')

    # set dummy training variables that are needed to initialize models but will not actually be used
    args.timesteps = 0
    args.learning_rate = 0
    args.discount_factor = 0
    args.number_of_steps = 1
    args.number_of_environments = 1
    args.batch_size = 1
    args.momentum = 0
    args.rmsp_decay = 0
    args.rmsp_epsilon = 0
    args.entropy_weight = 0
    args.critic_weight = 0
    args.max_grad_norm = 0
    args.sampling_method = 'max'
    args.epsilon = 0

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
