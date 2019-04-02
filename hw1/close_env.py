#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:59:26 2019

@author: zhaoxuanzhu
"""

# override env.close()
def env_wrapper(env):
    def close(self):
        if self.viewer is not None:
            glfw.terminate()
            self.viewer = None

    env.unwrapped.close = MethodType(close, env.unwrapped)
    return env


def close():
    env = env_wrapper(gym.make("Reacher-v2"))
    env.reset()
    for _ in range(50):
        env.render()
        env.step([0, 0])
    env.close()