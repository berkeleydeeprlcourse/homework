import numpy as np
import tensorflow as tf
import gym
from gym.envs.registration import registry
registry.env_specs['BipedalWalker-v2'].max_episode_steps = 100    # Time save ?
from gym.envs.box2d.bipedal_walker import BipedalWalker


class BipedalWalkerEnv(BipedalWalker):
    def __init__(self):
        BipedalWalker.__init__(self)
        obs_dim = self.observation_space.shape[0] - 10 - 2  # 10 lidar readings, 2 discrete values
        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self._max_episode_steps = 100

    def step(self, action):
        s, r, done, dict_ = BipedalWalker.step(self, action)
        s = s[:14]
        s = np.concatenate([s[:8], s[9:13]], -1)
        return s, r, done, dict_

    @staticmethod
    def cost_fn(states, actions, next_states):
        is_tf = tf.contrib.framework.is_tensor(states)
        is_single_state = (len(states.get_shape()) == 1) if is_tf else (len(states.shape) == 1)

        if is_single_state:
            states = states[None, ...]
            actions = actions[None, ...]
            next_states = next_states[None, ...]

        scores = tf.zeros(actions.get_shape()[0].value) if is_tf else np.zeros(actions.shape[0])

        heading_penalty_factor = 10

        # don't move both legs to the same direction
        leg1_angle = states[:, 4]
        leg2_angle = states[:, 8]
        if is_tf:
            scores += tf.cast(leg1_angle * leg2_angle > 0, tf.float32) * heading_penalty_factor
        else:
            scores += (leg1_angle * leg2_angle > 0) * heading_penalty_factor

        # Small plus to effort
        scores -= tf.abs(next_states[:, 5] - states[:, 5]) / 0.01
        scores -= tf.abs(next_states[:, 7] - states[:, 7]) / 0.01
        scores -= tf.abs(next_states[:, 9] - states[:, 9]) / 0.01
        scores -= tf.abs(next_states[:, 11] - states[:, 11]) / 0.01

        # Enforce to move only one leg
        pass

        if is_single_state:
            scores = scores[0]

        return scores
