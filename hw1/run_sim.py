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


class Object(object):
    pass


def behavior_cloning(obs, model, norm_params):
    feature_names = ['f{}'.format(i) for i in range(obs.size)]
    df = pd.DataFrame([obs], columns=feature_names)
    pred = bc.predict(df, model, norm_params)
    return pred


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def run_sim(envname, max_timesteps, num_rollouts, render, policy_fn, perf):
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
            action = policy_fn(obs)
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

    return {'observations': np.array(observations), 'actions': np.array(actions)}


def run_expert(expert_policy_file, envname, max_timesteps, num_rollouts, render, save, output, perf):
    print('loading and building expert policy')
    expert_policy = load_policy.load_policy(expert_policy_file)

    def expert_policy_fn(obs): return expert_policy(obs[None, :])

    with tf.Session():
        tf_util.initialize()
        perf['expert'] = {'mean': [], 'std': []}
        expert_data = run_sim(envname, max_timesteps,
                              num_rollouts, render, expert_policy_fn, perf['expert'])

        if save:
            expert_data_dir = os.path.join(output, 'expert_data')
            if not os.path.exists(expert_data_dir):
                os.makedirs(expert_data_dir)
            with open(os.path.join(expert_data_dir, '{}.pkl'.format(envname)), 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
            print('expert data saved!')


def run_model(ckpt, norm_params, envname, max_timesteps, num_rollouts, render, output, perf, perf_key, itr=None):
    print('loading and building behavior cloning policy')
    model, norm_params = bc.load(ckpt, norm_params)

    def cloning_policy_fn(obs): return behavior_cloning(obs, model, norm_params)
    perf[perf_key] = {'mean': [], 'std': []}
    data = run_sim(envname, max_timesteps, num_rollouts, render, cloning_policy_fn, perf[perf_key])
    if itr is not None:
        data_dir = os.path.join(output, 'dagger')
        with open(os.path.join(data_dir, '{}-{}-{}.pkl'.format(envname, 'bc', itr)), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('model data saved!')


def get_labels(expert_policy_file, raw_data_dir):
    expert_policy = load_policy.load_policy(expert_policy_file)

    def policy_fn(obs): return expert_policy(obs[None, :])

    # load observations from disk
    with open(raw_data_dir, 'rb') as fp:
        data = pickle.load(fp)
    observations = data['observations']

    # Get labels for each observation
    with tf.Session():
        tf_util.initialize()
        actions = []
        for obs in observations:
            action = policy_fn(obs)
            actions.append(action.squeeze())

    return {'observations': np.array(observations), 'actions': np.array(actions)}


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
    parser.add_argument('--dagger-itrs', type=int, help='num of dagger agg iterations')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint file')
    parser.add_argument('--norm-params', type=str, help='path to norm params file')
    parser.add_argument('--save-expert-policy', action='store_true')
    parser.add_argument('--save-perf-plots', action='store_true')
    args = parser.parse_args()

    perf = {}
    if args.expert:
        run_expert(args.expert_policy_file, args.envname, args.max_timesteps, args.num_rollouts,
                   args.render, args.save_expert_policy, args.output, perf)
    if args.cloning:
        run_model(args.checkpoint, args.norm_params, args.envname, args.max_timesteps,
                  args.num_rollouts, args.render, args.output, perf, 'cloning')
    if args.dagger:
        print('loading and building dagger policy')

        # collect initial expert data (D)
        run_expert(args.expert_policy_file, args.envname, args.max_timesteps, args.num_rollouts,
                   args.render, True, args.output, perf)
        train_args = Object()
        train_args.epochs = 1000
        train_args.test_size = 0.05
        dagger_itr = 0
        basedir = os.path.join(args.output, 'dagger')
        os.makedirs(basedir)
        while dagger_itr < args.dagger_itrs:
            # train model on D using latest and greatest dataset
            if dagger_itr:
                train_args.checkpoint_name = '{}-{}'.format(args.envname, dagger_itr - 1)
                last_agg_data_dir = os.path.join(basedir, '{}-{}.pkl'.format(args.envname,
                                                                             dagger_itr - 1))
                train_args.data_dir = last_agg_data_dir
            else:
                train_args.checkpoint_name = '{}-{}'.format(args.envname, 0)
                train_args.data_dir = os.path.join(os.path.join(args.output, 'expert_data'),
                                                   '{}.pkl'.format(args.envname))
            train_args.output = os.path.join(args.output, 'model-{}'.format(dagger_itr))
            checkpoint_dir = os.path.join(train_args.output, 'checkpoints')
            bc.train(train_args)

            # run model and get observations
            ckpt = '{}.h5'.format(os.path.join(checkpoint_dir, train_args.checkpoint_name))
            norm_params = '{}.pkl'.format(os.path.join(checkpoint_dir, train_args.checkpoint_name))
            run_model(ckpt, norm_params, args.envname, args.max_timesteps, args.num_rollouts,
                      args.render, args.output, perf, 'dagger-{}'.format(dagger_itr), dagger_itr)

            # run expert policy on the model generated observations
            data_dir = os.path.join(basedir, '{}-{}-{}.pkl'.format(args.envname, 'bc', dagger_itr))
            labeled_data = get_labels(args.expert_policy_file, data_dir)
            print('extracted labeled data: {} obs and {} actions'
                  .format(labeled_data['observations'].shape, labeled_data['actions'].shape))

            # D = D + D'
            # last dataset from disk
            with open(train_args.data_dir , 'rb') as fp:
                last_dataset = pickle.load(fp)
                print('loaded dataset: {} obs and {} actions'
                      .format(last_dataset['observations'].shape, last_dataset['actions'].shape))
            # merge latest and greatest with labeled_obs
            aggregared_dataset = {}
            aggregared_dataset['observations']=np.concatenate([last_dataset['observations'],labeled_data['observations']])
            aggregared_dataset['actions']=np.concatenate([last_dataset['actions'],labeled_data['actions']])
            print('aggregared_dataset dataset: {} obs and {} actions'.format(
                aggregared_dataset['observations'].shape, aggregared_dataset['actions'].shape))
            # save merged datasets to disk
            agg_data_dir = os.path.join(basedir, '{}-{}.pkl'.format(args.envname, dagger_itr))
            with open(agg_data_dir, 'wb') as f:
                pickle.dump(aggregared_dataset, f, pickle.HIGHEST_PROTOCOL)
            print('aggregared data saved!')

            # increase dagger iteration
            dagger_itr += 1

    if args.save_perf_plots:
        plt.xlabel('mean')
        plt.ylabel('standard deviation')
        i = 0
        markers = ['^', 'v', '>', '<', '.', 's', '+', 'D', '8', 'p', 'x', 'd', '1']
        for policy_name, policy_perf in perf.items():
            mean = policy_perf['mean']
            std = policy_perf['std']
            plt.errorbar(mean, std, std, marker=markers[i], label=policy_name)
            i += 1

        plt.legend()
        plots_dir = os.path.join(args.output, 'plots')
        os.makedirs(plots_dir)
        plt.savefig(os.path.join(plots_dir, '{}.png'.format(args.envname)))
        plt.close('all')


if __name__ == '__main__':
    main()
