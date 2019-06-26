import os
from baselines.bench import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import FrameStack
from baselines.common.retro_wrappers import Downsample, Rgb2gray
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines import logger
import gym
import os
import vizdoomgym
from vizdoomgym.envs.vizdoomenv import VizdoomEnv


def make_test_env(config):

    def make():
        env = gym.make(config.env)
        if config.downscale_ratio != 1:
            env = Downsample(env, config.downscale_ratio)
        if config.grayscale:
            env = Rgb2gray(env)
        if config.framestacking != 0:
            env = FrameStack(env, config.framestacking)
        if config.seed is not None:
            env.seed(config.seed)
        return env

    env = DummyVecEnv([make])
    if config.save_video:
        env = VecVideoRecorder(env, config.video_path, _save_test_video_when, video_length=1000000000)
    return env


# returns false since a video will be recorded at every reset anyway
def _save_test_video_when():
    return False


def make_train_env(config):
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'
    logger.configure(config.log_path)

    def make_thunk(rank):
        return lambda: make_train_sub_env(config, rank)

    sub_envs = [make_thunk(i) for i in range(config.number_of_environments)]
    env = SubprocVecEnv(sub_envs)
    if config.save_video:
        env = VecVideoRecorder(env, config.video_path, _save_train_video_when(), video_length=300)
    return env


def make_train_sub_env(config, subrank):
    env = gym.make(config.env)
    if isinstance(env, VizdoomEnv):
        env.set_skipcount(config.skipcount)
    if config.downscale_ratio != 1:
        env = Downsample(env, config.downscale_ratio)
    if config.grayscale:
        env = Rgb2gray(env)
    if config.framestacking != 0:
        env = FrameStack(env, config.framestacking)
    if config.seed is not None:
        env.seed(config.seed + subrank)
    sub_log_dir = os.path.join(config.log_path, str(subrank))
    env = Monitor(env, sub_log_dir, allow_early_resets=True)
    return env


def _save_train_video_when():
    next_t = 1

    def _save_when(t):
        nonlocal next_t
        if next_t <= t:
            next_t *= 2
            return t > 50000
        return False

    return _save_when
