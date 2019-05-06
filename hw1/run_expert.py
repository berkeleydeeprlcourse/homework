#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    # Using MuJoCo
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render --num_rollouts 20

    # Using Roboschool
    python run_expert.py experts/RoboschoolHumanoid-v1.py 'RoboschoolHumanoid-v1' --engine Roboschool --render --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)

Additional roboschool models were added with policies from the Roboschool agent zoo (https://github.com/openai/roboschool).
This code was rebased over Alex Hofer <rofer@google.com> code.
"""
import argparse
import importlib
import os
import pickle
import tensorflow as tf
import numpy as np
import gym
import tf_util
import load_policy


EXPERT_DIR = "experts"
ROBOSCOOL_EXPERT_DATA_DIR = 'hw1/expert_data'
SUPERVISED_MODELD_DATA_DIR = 'hw1/supervised_modeled_data'
ROBOSCOOL_AVAILABLE_ENVS = ['RoboschoolAnt-v1', 'RoboschoolHumanoid-v1', 'RoboschoolHalfCheetah-v1', 'RoboschoolReacher-v1',
                  'RoboschoolHopper-v1', 'RoboschoolWalker2d-v1']


ROBOSCHOOL_ENGINE = 'Roboschool'
MOJOCO_ENGINE = 'MuJoCo'
ENGINES = [ROBOSCHOOL_ENGINE, MOJOCO_ENGINE]


def run_mojoco_policy(expert_policy_file, num_rollouts, envname, max_timesteps=None, render=False, verbose=True):
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
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': np.array(returns)}
        return expert_data


def run_policy(env, policy, num_rollouts, description, max_timesteps=None, render=False, verbose=True):
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        if verbose:
            print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy.act(obs)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0 and verbose: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)


    print('Env description:', description)
    #print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions),
                   'returns': np.array(returns)}
    return expert_data


def run_expert_policy(num_rollouts, envname, max_timesteps=None, render=False, verbose=True):
    assert envname in ROBOSCOOL_AVAILABLE_ENVS
    # Load the policy module
    module_name = "%s.%s" % (EXPERT_DIR, envname)
    policy_module = importlib.import_module(module_name)

    env, policy = policy_module.get_env_and_policy()
    description = "Expert policy for module %s" % envname
    return run_policy(env=env, policy=policy, num_rollouts=num_rollouts, description=description,
                             max_timesteps=max_timesteps, render=render, verbose=verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--engine', type=str, default=MOJOCO_ENGINE)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    if args.engine == ROBOSCHOOL_ENGINE:
        print('loading %s expert policy' % ROBOSCHOOL_ENGINE)
        expert_data = run_expert_policy(num_rollouts=args.num_rollouts, envname=args.envname, max_timesteps=args.max_timesteps, render=args.render, verbose=True)

    else:
        print('loading %s expert policy' % MOJOCO_ENGINE)
        expert_data = run_mojoco_policy(expert_policy_file=args.expert_policy_file, num_rollouts=args.num_rollouts,
                                        envname=args.envname, max_timesteps=args.max_timesteps, render=args.render)

    returns = expert_data['returns']
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
