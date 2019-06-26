import numpy as np
import pandas as pd


def make_rollouts(config, env, model):
    rollouts = pd.DataFrame(columns=[
        'reward',
        'length',
        'ammo_gained',
        'ammo_lost',
        'health_gained',
        'health_lost',
        'deaths',
        'frags',
        'kills',
        'accuracy',
        'hits_given',
        'hits_taken',
    ])
    for i in range(config.rollouts):
        ob = env.reset()
        done = False
        length = 0
        total_reward = 0
        while not done:
            action = model.step(ob)
            action = action[0]
            ob, reward, done, info = env.step(action)
            length += 1
            total_reward += reward[0]
        info = info[0]
        accuracy = safe_divide((info['hits_given'], info['ammo_lost']))
        rollouts.loc[i] = [
            total_reward,
            length,
            info['ammo_gained'],
            info['ammo_lost'],
            info['health_gained'],
            info['health_lost'],
            info['deaths'],
            info['frags'],
            info['kills'],
            accuracy,
            info['hits_given'],
            info['hits_taken'],
        ]
    return rollouts


def is_not_none(x):
    return x is not None


def safe_divide(values):
    a, b = values
    if b == 0:
        return None
    return a / b


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
