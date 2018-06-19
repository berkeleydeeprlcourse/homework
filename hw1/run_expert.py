#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def generate_all_rollout_data():
    generate_rollout_data('experts/HalfCheetah-v1.pkl', 'HalfCheetah-v1', 500, 1000, False, 'data/' )
    generate_rollout_data('experts/Hopper-v1.pkl', 'Hopper-v1', 500, 1000, False, 'data/' )
    generate_rollout_data('experts/Ant-v1.pkl', 'Ant-v1', 500, 1000, False, 'data/' )
    generate_rollout_data('experts/Humanoid-v1.pkl', 'Humanoid-v1', 500, 1000, False, 'data/' )
    generate_rollout_data('experts/Reacher-v1.pkl', 'Reacher-v1', 500, 1000, False, 'data/' )
    generate_rollout_data('experts/Walker2d-v1.pkl', 'Walker2d-v1', 500, 1000, False, 'data/' ) 


def generate_rollout_data(expert_policy_file, env_name, max_timesteps, num_rollouts, render, output_dir):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(env_name)
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
        
        if output_dir is not 'None':
            print(output_dir)
            filename = '{}_data_{}_rollouts_{}_timesteps.pkl'.format(env_name, num_rollouts, max_timesteps)
            with open(output_dir + filename,'wb') as f:
                pickle.dump([np.array(observations), np.array(actions)], f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--output_dir", type=str, default='data/')
    args = parser.parse_args()

    generate_rollout_data(args.expert_policy_file, args.envname, args.max_timesteps, args.num_rollouts, args.render, args.output_dir)
