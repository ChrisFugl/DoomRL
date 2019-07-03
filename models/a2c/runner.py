import numpy as np


'''
class used to generates a batch of experiences
'''
class Runner:
    def __init__(self, env, model, config):

        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        self.batch_ob_shape = (config.batch_size, nh, nw, nc)
        self.obs = self.env.reset()
        self.gamma = config.discount_factor
        self.nsteps = config.number_of_steps
        self.dones = [False for _ in range(config.number_of_environments)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_infos = [], [], [], [], [], []

        for n in range(self.nsteps):
            # Given observations, take action and calculate Value (V(s))
            actions, values = self.model.step(self.obs)

            # append experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            #take next action, observe rewards
            obs, rewards, dones, infos = self.env.step(actions)

            for info in infos:
                episode = info.get('episode')
                if episode:
                    mb_infos.append({
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

            self.dones = dones
            self.obs = obs

            #check if episode is over
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0

            mb_rewards.append(rewards)

        mb_dones.append(self.dones)

        # convert batch of steps in different environments to batch of rollouts
        # from shape (steps, envs) to (envs, steps)
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        #get last values
        last_values = self.model.value(self.obs).tolist()

        # discount
        if self.gamma > 0:
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = self.discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = self.discount_with_dones(rewards, dones, self.gamma)
                mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        return mb_obs, mb_rewards, mb_masks, mb_actions, mb_values, mb_infos

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]
