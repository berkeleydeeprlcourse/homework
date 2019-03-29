#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import load_policy
import model as bc
import tf_util


def behavior_cloning(obs, model, norm_params):
    feature_names = ['f{}'.format(i) for i in range(obs.size)]
    df = pd.DataFrame([obs], columns=feature_names)
    pred = bc.predict(df, model, norm_params)
    return pred


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def run_sim(envname, max_timesteps, num_rollouts, render, policy_fn, perf, expert_policy_fn=None):
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    expert_actions = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs)
            if expert_policy_fn:
                expert_action = expert_policy_fn(obs)
                expert_actions.append(expert_action.squeeze())
            observations.append(obs)
            actions.append(action.squeeze())
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    perf['mean'].append(np.mean(returns))
    perf['std'].append(np.std(returns))

    return {'observations': np.array(observations), 'actions': np.array(actions)},
        {'observations': np.array(observations), 'expert_actions': np.array(expert_actions)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--output', type=str, required=True, help='path to output artifacts dir')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num-rollouts', type=int, default=20, help='number of expert roll outs')
    parser.add_argument('--expert', action='store_true')
    parser.add_argument('--cloning', action='store_true')
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--dagger-itrs', type=int, nargs='+', help='num of dagger agg iterations')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint file')
    parser.add_argument('--norm-params', type=str, help='path to norm params file')
    parser.add_argument('--save-expert-policy', action='store_true')
    parser.add_argument('--save-perf-plots', action='store_true')
    args = parser.parse_args()

    perf = {}

    if args.expert:
        print('loading and building expert policy')
        expert_policy = load_policy.load_policy(args.expert_policy_file)

        def expert_policy_fn(obs): return expert_policy(obs[None, :])

        with tf.Session():
            tf_util.initialize()
            perf['expert'] = {'mean': [], 'std': []}
            expert_data, _ = run_sim(args.envname, args.max_timesteps,
                                  args.num_rollouts, args.render, expert_policy_fn, perf['expert'])

            if args.save_expert_policy:
                expert_data_dir = os.path.join(args.output, 'expert_data')
                os.makedirs(expert_data_dir)
                with open(os.path.join(expert_data_dir, '{}.pkl'.format(args.envname)), 'wb') as f:
                    pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
                print('expert data saved!')

    if args.cloning:
        print('loading and building behavior cloning policy')
        model, norm_params = bc.load(args.checkpoint, args.norm_params)

        def cloning_policy_fn(obs): return behavior_cloning(obs, model, norm_params)
        perf['cloning'] = {'mean': [], 'std': []}
        run_sim(args.envname, args.max_timesteps, args.num_rollouts, args.render,
                cloning_policy_fn, perf['cloning'])

    if args.dagger:
        print('loading and building dagger policy')

        # collect initial expert data (D)

        # while num iterations
        # train model on D

        # use model to generate new set of obs with expert actions for each obs (D')

        # D = D + D'

    if args.save_perf_plots:
        plt.xlabel('mean')
        plt.ylabel('standard deviation')
        i = 0
        for policy_name, policy_perf in perf.items():
            mean = policy_perf['mean']
            std = policy_perf['std']
            plt.errorbar(mean, std, std, marker='^', label=policy_name)
            i += 1

        plt.legend()
        plots_dir = os.path.join(args.output, 'plots')
        os.makedirs(plots_dir)
        plt.savefig(os.path.join(plots_dir, '{}.png'.format(args.envname)))


if __name__ == '__main__':
    main()
