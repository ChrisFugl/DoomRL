import gym
from gym import spaces
from vizdoom import *
import numpy as np
import os
from gym.envs.classic_control import rendering

COLLECT_VARIABLES = [
    ('dead', GameVariable.DEAD),
    ('frags', GameVariable.FRAGCOUNT),
    ('health', GameVariable.HEALTH),
    ('hits_given', GameVariable.HITCOUNT),
    ('hits_taken', GameVariable.HITS_TAKEN),
    ('kills', GameVariable.KILLCOUNT),
]

CONFIGS = [['basic.cfg', 3],                # 0
           ['deadly_corridor.cfg', 7],      # 1
           ['defend_the_center.cfg', 3],    # 2
           ['defend_the_line.cfg', 3],      # 3
           ['health_gathering.cfg', 3],     # 4
           ['my_way_home.cfg', 5],          # 5
           ['predict_position.cfg', 3],     # 6
           ['take_cover.cfg', 2],           # 7
           ['deathmatch.cfg', 20],          # 8
           ['health_gathering_supreme.cfg', 3]]  # 9


class VizdoomEnv(gym.Env):

    def __init__(self, level):

        # init game
        self.game = DoomGame()
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, CONFIGS[level][0]))
        self.game.set_window_visible(False)
        self.game.init()
        self.state = None
        self.skipcount = 4

        self.action_space = spaces.Discrete(CONFIGS[level][1])
        self.observation_space = spaces.Box(0, 255, (self.game.get_screen_height(),
                                                     self.game.get_screen_width(),
                                                     self.game.get_screen_channels()),
                                            dtype=np.uint8)
        self.viewer = None

    def set_skipcount(self, skipcount):
        self.skipcount = skipcount

    def step(self, action):
        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        reward = self.game.make_action(act, self.skipcount)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        game_info = {k: self.game.get_game_variable(v) for (k, v) in COLLECT_VARIABLES}

        self.info['frags'] = game_info['frags']
        self.info['hits_given'] = game_info['hits_given']
        self.info['hits_taken'] = game_info['hits_taken']
        self.info['kills'] = game_info['kills']

        ammo = self._get_total_ammo()
        ammo_delta = ammo - self.ammo
        health_delta = game_info['health'] - self.health
        if ammo_delta < 0:
            self.info['ammo_lost'] += abs(ammo_delta)
        else:
            self.info['ammo_gained'] += ammo_delta
        if health_delta < 0:
            self.info['health_lost'] += abs(health_delta)
        else:
            self.info['health_gained'] += health_delta
        if game_info['dead']:
            self.info['deaths'] += 1

        self.ammo = ammo
        self.health = game_info['health']

        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            info = {}
        else:
            observation = np.uint8(np.zeros(self.observation_space.shape))
            info = self.info.copy()

        return observation, reward, done, info

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        img = self.state.screen_buffer
        self.ammo = self._get_total_ammo()
        self.health = self.game.get_game_variable(GameVariable.HEALTH)
        self.info = {
            'ammo_gained': 0,
            'ammo_lost': 0,
            'deaths': 0,
            'frags': 0,
            'health_gained': 0,
            'health_lost': 0,
            'hits_given': 0,
            'hits_taken': 0,
            'kills': 0,
        }
        return np.transpose(img, (1, 2, 0))

    def render(self, mode='human'):
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
        except AttributeError:
            pass
        return img

    @staticmethod
    def get_keys_to_action():
        # you can press only one key at a time!
        keys = {(): 2,
                (ord('a'),): 0,
                (ord('d'),): 1,
                (ord('w'),): 3,
                (ord('s'),): 4,
                (ord('q'),): 5,
                (ord('e'),): 6}
        return keys

    def _get_total_ammo(self):
        return self.game.get_game_variable(GameVariable.AMMO0) \
            + self.game.get_game_variable(GameVariable.AMMO1) \
            + self.game.get_game_variable(GameVariable.AMMO2) \
            + self.game.get_game_variable(GameVariable.AMMO3) \
            + self.game.get_game_variable(GameVariable.AMMO4) \
            + self.game.get_game_variable(GameVariable.AMMO5) \
            + self.game.get_game_variable(GameVariable.AMMO6) \
            + self.game.get_game_variable(GameVariable.AMMO7) \
            + self.game.get_game_variable(GameVariable.AMMO8) \
            + self.game.get_game_variable(GameVariable.AMMO9)
