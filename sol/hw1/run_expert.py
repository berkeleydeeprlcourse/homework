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
import random
from keras.models import Sequential
from keras.layers import Dense

# Extracts the number of input and output units from an OpenAI Gym environment.
def env_dims(env):
    return (env.observation_space.shape[0], env.action_space.shape[0])
    
# A neural network that learns a mapping from observations to actions.
class SupervisedPolicy:
    def __init__(self, env):
        input_len, output_len = env_dims(env)
        
        self.model = Sequential()
        self.model.add(Dense(units=64, input_dim=input_len, activation='relu'))
        self.model.add(Dense(units=output_len))
        
        self.model.compile(loss='mse', optimizer='sgd')
        
    def train(self, train_data, val_data, epochs, verbose):
        self.model.fit(train_data[0], train_data[1],
                  batch_size=128,
                  epochs=epochs,
                  verbose=verbose,
                  validation_data=val_data)
                  
    def act(self, obs):
        obs_batch = np.expand_dims(obs, 0)
        act_batch = self.model.predict_on_batch(obs_batch)
        return np.ndarray.flatten(act_batch)
    
    def save(self, filename):
        self.model.save_weights(filename)
    
    def load(self, filename):
        self.model.load_weights(filename)

# A policy that may use the student's action but, with probability
# fraction_assist, uses the teacher's action instead. In either case, it
# remembers the teacher's action, which can be used for supervised learning.
class AssistedPolicy:
    def __init__(self, env, student, teacher):
        self.CAPACITY = 50000
        self.student = student
        self.teacher = teacher
        self.fraction_assist = 1.
        self.next_idx = 0
        self.size = 0
        
        input_len, output_len = env_dims(env)
        self.obs_data = np.empty([self.CAPACITY, input_len])
        self.act_data = np.empty([self.CAPACITY, output_len])
    
    def act(self, obs):
        teacher_act = self.teacher.act(obs)
        self.obs_data[self.next_idx] = obs
        self.act_data[self.next_idx] = teacher_act
        self.next_idx = (self.next_idx + 1) % self.CAPACITY
        self.size = min(self.size + 1, self.CAPACITY)
        
        if random.random() < self.fraction_assist:
            return teacher_act
        else:
            return self.student.act(obs)
    
    def teacher_data(self):
        return (self.obs_data[:self.size], self.act_data[:self.size])

# Generates rollouts of the policy on the environment, prints the mean & std of
# the rewards, and returns the observations and actions.
def generate_rollouts(env, policy, max_steps, num_rollouts, render, verbose):
    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
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
            if steps % 100 == 0 and verbose >= 2:
                print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        if verbose >= 1:
            print('rollout %i/%i return=%f' % (i + 1, num_rollouts, totalr))
        returns.append(totalr)
        
    print('Return summary: mean=%f, std=%f' % (np.mean(returns), np.std(returns)))

    return (np.array(observations), np.array(actions))

# Make a small but low-variance validation test by subsampling across many episodes.
def make_validation(env, teacher, max_steps):
    val_data = generate_rollouts(env, teacher, max_steps, 50, False, 0)
    val_data = (val_data[0][::10], val_data[1][::10])
    return val_data

# Trains the student network using Behavior Cloning.
def behavior_cloning(env, student, teacher, max_steps, verbose):
    train_data = generate_rollouts(env, teacher, max_steps, 100, False, 0)
    val_data = make_validation(env, teacher, max_steps)
    student.train(train_data, val_data, 300, verbose)
    
    return student

# Trains the student network using DAgger.
def dagger(env, student, teacher, max_steps, verbose):
    val_data = make_validation(env, teacher, max_steps)
    mixed_policy = AssistedPolicy(env, student, teacher)
    
    for i in range(200):
        print('DAgger iter', i)
        if i == 0:
            rollouts = 50
            epochs = 100
        else:
            rollouts = 1
            epochs = 4
            mixed_policy.fraction_assist -= 0.01
        
        generate_rollouts(env, mixed_policy, max_steps, rollouts, False, verbose)
        student.train(mixed_policy.teacher_data(), val_data, epochs, verbose)
    
    return student

# Specify and read arguments from the command line.
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    
    parser_algorithm = parser.add_mutually_exclusive_group()
    parser_algorithm.add_argument('--cloning', action='store_true')
    parser_algorithm.add_argument('--dagger', action='store_true')
    
    parser.add_argument('-v', '--verbose', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20)
    
    parser.add_argument('--load_weights', type=str)
    parser.add_argument('--save_weights', type=str)
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    print('loading expert policy')
    module_name = args.expert_policy_file.replace('/', '.')
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    policy_module = importlib.import_module(module_name)
    print('loaded')
    
    env, teacher = policy_module.get_env_and_policy()
    max_steps = args.max_timesteps or env.spec.timestep_limit
    student = SupervisedPolicy(env)
    
    if args.load_weights:
        student.load(args.load_weights)
    
    if args.cloning:
        behavior_cloning(env, student, teacher, max_steps, args.verbose)
    elif args.dagger:
        dagger(env, student, teacher, max_steps, args.verbose)
    
    if args.save_weights:
        student.save(args.save_weights)
    
    generate_rollouts(env, student, max_steps, args.num_rollouts, args.render, args.verbose)

if __name__ == '__main__':
    main()
