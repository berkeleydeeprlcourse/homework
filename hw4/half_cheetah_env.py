import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
            # self.get_body_comvel("torso").flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

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

        # dont move front shin back so far that you tilt forward
        front_leg = states[:, 5]
        my_range = 0.2
        if is_tf:
            scores += tf.cast(front_leg >= my_range, tf.float32) * heading_penalty_factor
        else:
            scores += (front_leg >= my_range) * heading_penalty_factor

        front_shin = states[:, 6]
        my_range = 0
        if is_tf:
            scores += tf.cast(front_shin >= my_range, tf.float32) * heading_penalty_factor
        else:
            scores += (front_shin >= my_range) * heading_penalty_factor

        front_foot = states[:, 7]
        my_range = 0
        if is_tf:
            scores += tf.cast(front_foot >= my_range, tf.float32) * heading_penalty_factor
        else:
            scores += (front_foot >= my_range) * heading_penalty_factor

        scores -= (next_states[:, 17] - states[:, 17]) / 0.01

        if is_single_state:
            scores = scores[0]

        return scores
