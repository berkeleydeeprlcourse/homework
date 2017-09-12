#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 20 --expert_data_filename expert_data/expert_data_Hopper-v1.pkl

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
import argparse

import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

from pickle_util import save_obj

def main(expert_policy_file,
         envname,
         num_rollouts,
         expert_data_filename=None,
         max_timesteps=None,
         render=False):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        if expert_data_filename:
            save_obj(expert_data, expert_data_filename)

    return returns

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('expert_policy_file', type=str)
  parser.add_argument('envname', type=str)
  parser.add_argument('--expert_data_filename', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--max_timesteps', type=int)
  parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
  args = parser.parse_args()

  main(args.expert_policy_file,
       args.envname,
       args.num_rollouts,
       args.expert_data_filename,
       args.max_timesteps,
       args.render)
