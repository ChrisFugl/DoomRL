from arguments import parse_train_config
from environment import make_train_env
import gym
from logger import Logger
import numpy as np
import os
import random
import tensorflow as tf


def main():
    config = parse_train_config()
    env = make_train_env(config)
    tb_path = os.path.join(config.log_path, 'tb')
    logger = Logger(tb_path)
    if config.seed is not None:
        set_seed(config.seed)
    algorithm = config.algorithm
    if algorithm == 'baseline_a2c':
        from models.baselines.a2c import train
    elif algorithm == 'baseline_acer':
        from models.baselines.acer import train
    elif algorithm == 'baseline_ppo':
        from models.baselines.ppo import train
    elif algorithm == 'a2c':
        from models.a2c import train
    elif algorithm == 'acer':
        from models.acer import train
    elif algorithm == 'ppo':
        from models.ppo import train
    else:
        raise Exception(f'Unknown algorithm: {algorithm}')
    train(config, env, logger)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    main()
