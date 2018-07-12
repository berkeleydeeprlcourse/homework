import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv

class HalfCheetahTorsoEnv(HalfCheetahEnv, utils.EzPickle):
    """
    Adds .get_body_com("torso").flat to observations and sets frame skip to 1
    """

    def __init__(self, **kwargs):
        # frame skip to 1
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 1)
        utils.EzPickle.__init__(self)


    def _get_obs(self):
        obs = np.concatenate([
            HalfCheetahEnv._get_obs(self),
            self.get_body_com("torso").flat,
        ])
        return obs