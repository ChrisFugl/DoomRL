from arguments import parse_test_config
from environment import make_test_env
import gym
import models
import numpy as np
import os
import random
import tensorflow as tf


def main():
    config = parse_test_config()
    env = make_test_env(config)
    if config.seed is not None:
        set_seed(config.seed)
    algorithm = config.algorithm
    if algorithm == 'baseline_a2c':
        from models.baselines.a2c import test
    elif algorithm == 'baseline_ppo':
        from models.baselines.ppo import test
    elif algorithm == 'a2c':
        from models.a2c import test
    elif algorithm == 'ppo':
        from models.ppo import test
    else:
        raise Exception(f'Unknown algorithm: {algorithm}')
    rollouts = test(config, env)
    save_path = os.path.join(config.out, 'rollouts.csv')
    rollouts.to_csv(save_path, index=None)


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
