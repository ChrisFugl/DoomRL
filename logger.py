import os
from models.utils import is_not_none, safe_divide, safe_mean
import tensorflow as tf


class Logger:

    def __init__(self, tb_path):
        os.makedirs(tb_path, exist_ok=True)
        self._writer = tf.summary.FileWriter(tb_path)

    def add_infos(self, info_buffer):
        self._add_to_summary('episode_means', 'reward', safe_mean([info['reward'] for info in info_buffer]))
        self._add_to_summary('episode_means', 'length', safe_mean([info['length'] for info in info_buffer]))

        ammo_lost = [info['ammo_lost'] for info in info_buffer]
        hits_given = [info['hits_given'] for info in info_buffer]
        accuracy = list(filter(is_not_none, map(safe_divide, zip(hits_given, ammo_lost))))

        self._add_to_summary('game_variables', 'accuracy', safe_mean(accuracy))
        self._add_to_summary('game_variables', 'ammo_gained', safe_mean([info['ammo_gained'] for info in info_buffer]))
        self._add_to_summary('game_variables', 'ammo_lost', safe_mean(ammo_lost))
        self._add_to_summary('game_variables', 'hits_given', safe_mean(hits_given))
        self._add_to_summary('game_variables', 'hits_taken', safe_mean([info['hits_taken'] for info in info_buffer]))
        self._add_to_summary('game_variables', 'health_gained', safe_mean([info['health_gained'] for info in info_buffer]))
        self._add_to_summary('game_variables', 'health_lost', safe_mean([info['health_lost'] for info in info_buffer]))
        self._add_to_summary('game_variables', 'deaths', safe_mean([info['deaths'] for info in info_buffer]))
        self._add_to_summary('game_variables', 'frags', safe_mean([info['frags'] for info in info_buffer]))
        self._add_to_summary('game_variables', 'kills', safe_mean([info['kills'] for info in info_buffer]))

    def add_value(self, key, value):
        self._summary.value.add(tag=key, simple_value=value)

    def start_summary(self):
        self._summary = tf.Summary()

    def log_summary(self, timestep):
        self._writer.add_summary(self._summary, timestep)
        self._writer.flush()

    def summary(self, timestep, fps, info_buffer, total_loss, policy_loss, value_loss, entropy):
        self.start_summary()
        self.add_infos(info_buffer)
        self.add_value('model/fps', fps)
        self.add_value('model/total_loss', total_loss)
        self.add_value('model/policy_loss', policy_loss)
        self.add_value('model/value_loss', value_loss)
        self.add_value('model/entropy', entropy)
        self.log_summary(timestep)

    def _add_to_summary(self, scope, name, value):
        self._summary.value.add(tag=f'{scope}/{name}', simple_value=value)
