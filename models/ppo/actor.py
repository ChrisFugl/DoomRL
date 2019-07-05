import numpy as np


class Actor:

    def __init__(self, config, env, model):
        self.config = config
        self.env = env
        self.model = model
        self.last_dones = [False for _ in range(config.number_of_environments)]
        self.last_observations = env.reset()
        self.gamma = config.discount_factor
        self.gae_lambda = config.gae_lambda

    def act(self):
        batch_observations = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_neg_log_p = []
        batch_dones = []
        batch_infos = []

        for _ in range(self.config.number_of_steps):
            actions, values, neg_log_p = self.model.step(self.last_observations)
            batch_observations.append(np.copy(self.last_observations))
            batch_actions.append(actions)
            batch_values.append(values)
            batch_neg_log_p.append(neg_log_p)
            batch_dones.append(self.last_dones)
            self.last_observations[:], rewards, self.last_dones, infos = self.env.step(actions)
            batch_rewards.append(rewards)

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

        batch_observations = np.asarray(batch_observations, dtype=self.last_observations.dtype)
        batch_actions = np.asarray(batch_actions, dtype=actions.dtype)
        batch_rewards = np.asarray(batch_rewards, dtype=rewards.dtype)
        batch_values = np.asarray(batch_values, dtype=values.dtype)
        batch_neg_log_p = np.asarray(batch_neg_log_p, dtype=neg_log_p.dtype)
        batch_dones = np.asarray(batch_dones, dtype=np.bool)

        # downscale rewards
        batch_rewards = batch_rewards / self.config.reward_scale

        # generalized advantage
        advantages = np.empty_like(batch_rewards)
        advantage = 0
        next_non_terminal = 1.0 - self.last_dones
        next_value = self.model.value(self.last_observations)
        for t in range(self.config.number_of_steps - 1, -1, -1):
            delta = batch_rewards[t] + self.gamma * next_non_terminal * next_value - batch_values[t]
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * advantage
            advantages[t] = advantage
            next_non_terminal = 1.0 - batch_dones[t]
            next_value = batch_values[t]

        batch_observations = self._reshape_batch(batch_observations)
        batch_actions = self._reshape_batch(batch_actions)
        batch_advantages = self._reshape_batch(advantages)
        batch_values = self._reshape_batch(batch_values)
        batch_neg_log_p = self._reshape_batch(batch_neg_log_p)

        return batch_observations, batch_actions, batch_advantages, batch_values, batch_neg_log_p, batch_infos

    def _reshape_batch(self, x):
        s = x.shape
        return x.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
