import numpy as np
import os
import tensorflow as tf


class Logger:
    def __init__(self, log_dir):
        self._summary_writer = tf.summary.FileWriter(
            os.path.expanduser(log_dir))

        self._rows = []

    def log_value(self, tag, value, step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self._summary_writer.add_summary(summary, step)

        self._rows.append("{tag:.<25} {value}".format(tag=tag, value=value))

    def log_values(self, dictionary, step):
        for tag, value in dictionary.items():
            self.log_value(tag, value, step)

    def flush(self):
        self._summary_writer.flush()
        print(format("", "_<25"))
        print("\n".join(self._rows))

        self._rows = []


class ReplayPool:
    def __init__(self, max_size, fields):
        max_size = int(max_size)
        self._max_size = max_size

        self.fields = {}
        self.field_names = []
        self.add_fields(fields)

        self._pointer = 0
        self._size = 0

    @property
    def size(self):
        return self._size

    def add_fields(self, fields):
        self.fields.update(fields)
        self.field_names += list(fields.keys())

        for field_name, field_attrs in fields.items():
            field_shape = [self._max_size] + list(field_attrs['shape'])
            initializer = field_attrs.get('initializer', np.zeros)
            setattr(self, field_name, initializer(field_shape))

    def _advance(self, count=1):
        self._pointer = (self._pointer + count) % self._max_size
        self._size = min(self._size + count, self._max_size)

    def add_sample(self, **kwargs):
        self.add_samples(1, **kwargs)

    def add_samples(self, num_samples=1, **kwargs):
        for field_name in self.field_names:
            idx = np.arange(self._pointer,
                            self._pointer + num_samples) % self._max_size
            getattr(self, field_name)[idx] = kwargs.pop(field_name)

        self._advance(num_samples)

    def random_indices(self, batch_size):
        if self._size == 0: return []
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(random_indices, field_name_filter)

    def batch_by_indices(self, indices, field_name_filter=None):
        field_names = self.field_names
        if field_name_filter is not None:
            field_names = [
                field_name for field_name in field_names
                if field_name_filter(field_name)
            ]

        return {
            field_name: getattr(self, field_name)[indices]
            for field_name in field_names
        }

    def get_statistics(self):
        return {
            'PoolSize': self._size,
        }


class SimpleReplayPool(ReplayPool):
    def __init__(self, observation_shape, action_shape, *args, **kwargs):
        self._observation_shape = observation_shape
        self._action_shape = action_shape

        fields = {
            'observations': {
                'shape': self._observation_shape,
                'dtype': 'float32'
            },
            # It's a bit memory inefficient to save the observations twice,
            # but it makes the code *much* easier since you no longer have
            # to worry about termination conditions.
            'next_observations': {
                'shape': self._observation_shape,
                'dtype': 'float32'
            },
            'actions': {
                'shape': self._action_shape,
                'dtype': 'float32'
            },
            'rewards': {
                'shape': [],
                'dtype': 'float32'
            },
            # self.terminals[i] = a terminal was received at time i
            'terminals': {
                'shape': [],
                'dtype': 'bool'
            },
        }

        super(SimpleReplayPool, self).__init__(*args, fields=fields, **kwargs)


class Sampler(object):
    def __init__(self, max_episode_length, prefill_steps):
        self._max_episode_length = max_episode_length
        self._prefill_steps = prefill_steps

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

        class UniformPolicy:
            def __init__(self, action_dim):
                self._action_dim = action_dim

            def eval(self, _):
                return np.random.uniform(-1, 1, self._action_dim)

        uniform_exploration_policy = UniformPolicy(env.action_space.shape[0])
        for _ in range(self._prefill_steps):
            self.sample(uniform_exploration_policy)

    def set_policy(self, policy):
        self.policy = policy

    def sample(self):
        raise NotImplementedError

    def random_batch(self, batch_size):
        return self.pool.random_batch(batch_size)

    def terminate(self):
        self.env.terminate()


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._episode_length = 0
        self._episode_return = 0
        self._last_episode_return = 0
        self._max_episode_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def sample(self, policy=None):
        policy = self.policy if policy is None else policy
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action = policy.eval(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)
        self._episode_length += 1
        self._episode_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observations=self._current_observation,
            actions=action,
            rewards=reward,
            terminals=terminal,
            next_observations=next_observation)

        if terminal or self._episode_length >= self._max_episode_length:
            self._current_observation = self.env.reset()
            self._episode_length = 0
            self._max_episode_return = max(self._max_episode_return,
                                           self._episode_return)
            self._last_episode_return = self._episode_return

            self._episode_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

    def get_statistics(self):
        statistics = {
            'MaxEpReturn': self._max_episode_return,
            'LastEpReturn': self._last_episode_return,
            'Episodes': self._n_episodes,
            'TimestepsSoFar': self._total_samples,
        }

        return statistics
