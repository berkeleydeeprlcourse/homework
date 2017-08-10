#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/RoboschoolHumanoid-v1.py --render \
            --num_rollouts 20
"""

import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import importlib
from keras.models import Sequential
from keras.layers import Dense, Activation

class BehaviorCloningPolicy:
    def __init__(self, train_data, val_data, epochs):
        input_len = train_data['observations'].shape[1]
        output_len = train_data['actions'].shape[1]

        self.model = Sequential()
        self.model.add(Dense(units=64, input_dim=input_len, activation='relu'))
        self.model.add(Dense(units=output_len))
        
        self.model.compile(loss='mse', optimizer='sgd')
                  
        self.model.fit(train_data['observations'], train_data['actions'],
                  epochs=epochs,
                  batch_size=128,
                  validation_data=(val_data['observations'], val_data['actions']))
                  
    def act(self, obs):
        obs_batch = np.expand_dims(obs, 0)
        act_batch = self.model.predict_on_batch(obs_batch)
        return np.ndarray.flatten(act_batch)
    
def simulate(env, policy, max_steps, num_rollouts, render):
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
            action = policy.act(obs)
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
    return expert_data
    
def behavior_cloning(env, teacher, max_steps):
    train_data = simulate(env, teacher, max_steps, 100, False)
    val_data = simulate(env, teacher, max_steps, 10, False)
    student = BehaviorCloningPolicy(train_data, val_data, 200)
    return student

def dagger(env, teacher, max_steps):
    train_data = simulate(env, teacher, max_steps, 20, False)
    val_data = simulate(env, teacher, max_steps, 10, False)
    
    for i in range(10):
        student = BehaviorCloningPolicy(train_data, val_data, 20)
        trace = simulate(env, student, max_steps, 10, False)
        actions = [teacher.act(obs) for obs in trace['observations']]
        train_data['observations'].append(trace['observations'])
        train_data['actions'].append(actions)
        # TODO:
        # - instead of .append, preallocate max space and use slices
        # - sometimes let the teacher act
        # - train incrementally, maintaining weights
        # - save and load model to/from a file
    
    return student

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading expert policy')
    module_name = args.expert_policy_file.replace('/', '.')
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    policy_module = importlib.import_module(module_name)
    print('loaded')
    
    env, teacher = policy_module.get_env_and_policy()
    max_steps = args.max_timesteps or env.spec.timestep_limit
    
    student = behavior_cloning(env, teacher, max_steps)
    #student = dagger(env, teacher, max_steps)
    
    simulate(env, student, max_steps, args.num_rollouts, args.render)

if __name__ == '__main__':
    main()
