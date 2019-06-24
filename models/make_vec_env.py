import os
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import FrameStack
from baselines.common.retro_wrappers import Downsample, Rgb2gray
from baselines.bench import Monitor
from baselines import logger
import gym
import vizdoomgym
from vizdoomgym.envs.vizdoomenv import VizdoomEnv


def make_vec_env(config):
    logger_dir = logger.get_dir()

    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            config,
            subrank=rank,
            logger_dir=logger_dir,
        )

    return SubprocVecEnv([make_thunk(i, initializer=None) for i in range(config.number_of_environments)])


def make_env(config, subrank=0, seed=None, logger_dir=None):
    env = get_env(config)
    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(subrank)),
                  allow_early_resets=True)
    return env

def get_env(config):
    env = gym.make(config.env)
    if isinstance(env, VizdoomEnv):
        env.set_skipcount(config.skipcount)
    if config.downscale_ratio != 1:
        env = Downsample(env, config.downscale_ratio)
    if config.grayscale:
        env = Rgb2gray(env)
    if config.framestacking != 0:
        env = FrameStack(env, config.framestacking)
    return env
