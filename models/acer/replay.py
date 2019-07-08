import numpy as np
from models.acer.utils import np_rollout2batch


class Replay:
    """
    Replay memory used to increase sample efficiency.
    """

    def __init__(self, config, env):
        action_dtype = env.action_space.dtype
        self._n_envs = config.number_of_environments
        self._n_steps = config.number_of_steps
        self._replay_start = config.replay_start // self._n_steps
        self._max_size = config.buffer_size // self._n_steps
        self._index = 0
        self._size = 0

        self._action_dtype = env.action_space.dtype
        self._n_actions = env.action_space.n
        self._observation_dtype = env.observation_space.dtype
        self._observation_shape = env.observation_space.shape
        observations_shape = (self._max_size, self._n_envs, self._n_steps, *self._observation_shape)
        self._observations = np.empty(observations_shape, dtype=self._observation_dtype)
        self._actions = np.empty((self._max_size, self._n_envs, self._n_steps,), dtype=self._action_dtype)
        self._rewards = np.empty((self._max_size, self._n_envs, self._n_steps,), dtype=np.float32)
        self._dones = np.empty((self._max_size, self._n_envs, self._n_steps,), dtype=np.bool)
        self._mus = np.empty((self._max_size, self._n_envs, self._n_steps, self._n_actions), dtype=np.float32)

    def _to_batch(self, indices):
        def _batch(x):
            rollout = np.empty((self._n_envs, *x.shape[2:]), dtype=x.dtype)
            for i in range(self._n_envs):
                rollout[i] = x[indices[i], i]
            return np_rollout2batch(rollout)

        return _batch

    def sample(self):
        indices = np.random.randint(low=0, high=self._size, size=self._n_envs)
        to_batch = self._to_batch(indices)
        return (
            to_batch(self._observations),
            to_batch(self._actions),
            to_batch(self._rewards),
            to_batch(self._dones),
            to_batch(self._mus)
        )

    def can_sample(self):
        return 0 < self._size and self._replay_start <= self._size

    def store(self, observations, actions, rewards, dones, mus):
        self._observations[self._index] = observations
        self._actions[self._index] = actions
        self._rewards[self._index] = rewards
        self._dones[self._index] = dones
        self._mus[self._index] = mus
        self._index = (self._index + 1) % self._max_size
        self._size = min(self._max_size, self._size + 1)
