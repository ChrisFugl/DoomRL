import numpy as np


class Actor:

    def __init__(self, config, env, model):
        self._action_dtype = env.action_space.dtype
        self._reward_scale = config.reward_scale
        self._n_steps = config.number_of_steps
        self._env = env
        self._model = model
        self._last_observations = env.reset()

    def act(self):
        """
        :returns: observations, actions, rewards, dones, mus, infos
        """
        batch_observations = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_mus = []
        batch_infos = []

        for _ in range(self._n_steps):
            batch_observations.append(np.copy(self._last_observations))
            actions, mus = self._model.step(self._last_observations)
            self._last_observations, rewards, dones, infos = self._env.step(actions)
            batch_actions.append(actions)
            batch_rewards.append(rewards / self._reward_scale)
            batch_dones.append(dones)
            batch_mus.append(mus)
            for info in infos:
                episode = info.get('episode')
                if episode:
                    batch_infos.append({
                        'reward': episode['r'],
                        'length': episode['l'],
                        'ammo_gained': info.get('ammo_gained'),
                        'ammo_lost': info.get('ammo_lost'),
                        'health_gained': info.get('health_gained'),
                        'health_lost': info.get('health_lost'),
                        'deaths': info.get('deaths'),
                        'frags': info.get('frags'),
                        'kills': info.get('kills'),
                        'hits_given': info.get('hits_given'),
                        'hits_taken': info.get('hits_taken'),
                    })

        # transform to make batch of (n_envs, n_steps, ...) rollouts
        batch_observations = self._to_batch(batch_observations, np.int)
        batch_actions = self._to_batch(batch_actions, self._action_dtype)
        batch_rewards = self._to_batch(batch_rewards, np.float)
        batch_dones = self._to_batch(batch_dones, np.bool)
        batch_mus = self._to_batch(batch_mus, np.float)

        return batch_observations, batch_actions, batch_rewards, batch_dones, batch_mus, batch_infos

    def _to_batch(self, x, dtype):
        x = np.array(x, dtype=dtype)
        s = x.shape
        return np.swapaxes(x, 0, 1)
