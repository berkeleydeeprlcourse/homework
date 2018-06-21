from __future__ import print_function
import os
import logging
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import load_policy

from data.bc_data import Data
from models.bc_model import Model

def config_logging(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def create_model(session, obs_samples, num_observations, num_actions, logger, optimizer, learning_rate, restore, checkpoint_dir):
    model = Model(obs_samples, num_observations, num_actions, checkpoint_dir, logger, optimizer, learning_rate)

    if restore:
        model.load(session)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model

def gather_expert_experience(num_rollouts, env, policy_fn, max_steps):
    with tf.Session():
        returns = []
        observations = []
        actions = []
        for i in tqdm(range(num_rollouts)):
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
                if steps >= max_steps:
                    break
            returns.append(totalr)

        expert_data = {'observations': np.stack(observations, axis=0),
                       'actions': np.squeeze(np.stack(actions, axis=0)),
                       'returns':np.array(returns)}
        return expert_data


def bc(expert_data_file, num_rollouts, num_epochs, optimizer, learning_rate, env_name, batch_size, restore, results_dir, max_timesteps=None):
    tf.reset_default_graph()
    
    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    logger.debug('loading and building expert policy')
    expert_policy_fn = load_policy.load_policy(expert_data_file)

    logger.debug('gather experience...')
    data = gather_expert_experience(num_rollouts, env, expert_policy_fn, max_steps)
    logger.info("Expert's reward mean : %f(%f)"%(np.mean(data['returns']),np.std(data['returns'])))

    # TODO: Split into train and validation sets, run validation and save model
    # data = Data(data, train_ratio=0.9, val_ratio=0.05)

    num_samples = len(data['observations'])

    with tf.Session() as session:
        model = create_model(session, data['observations'], data['observations'].shape[1], data['actions'].shape[1], logger, optimizer, learning_rate, restore, results_dir)

        for epoch in tqdm(range(num_epochs)):
            perm = np.random.permutation(data['observations'].shape[0])

            obs_samples = data['observations'][perm]
            action_samples = data['actions'][perm]

            loss = 0.
            for k in range(0,obs_samples.shape[0], batch_size):
                loss += model.update(session, obs_samples[k:k+batch_size],
                                     action_samples[k:k+batch_size])
        
            new_exp = model.test_run(session, env, max_steps )
            tqdm.write("Epoch %3d Loss %f Reward %f" %(epoch, loss/num_samples, new_exp['reward']))

        env = wrappers.Monitor(env, results_dir, force=True)

        results = []
        for _ in tqdm(range(10)):
            results.append(model.test_run(session, env, max_steps )['reward'])
        logger.info("Reward mean & std of Cloned policy: %f(%f)"%(np.mean(results), np.std(results)))
    return np.mean(data['returns']), np.std(data['returns']), np.mean(results), np.std(results)

def dagger(expert_data_file, env_name, optimizer, learning_rate, restore, results_dir,
        num_rollouts=10, max_timesteps=None, num_epochs=100, batch_size=32, save=None):
    tf.reset_default_graph()

    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    logger.debug('loading and building expert policy')
    expert_policy_fn = load_policy.load_policy(expert_data_file)
    logger.debug('gather experience...')
    data = gather_expert_experience(num_rollouts, env, expert_policy_fn, max_steps)
    logger.info("Expert's reward mean : %f(%f)"%(np.mean(data['returns']), np.std(data['returns'])))
    logger.debug('building cloning policy')

    with tf.Session() as session:
        model = create_model(session, data['observations'], data['observations'].shape[1], data['actions'].shape[1], logger, optimizer, learning_rate, restore, results_dir)

        for epoch in tqdm(range(num_epochs)):
            num_samples = data['observations'].shape[0]
            perm = np.random.permutation(num_samples)

            obsv_samples = data['observations'][perm]
            action_samples = data['actions'][perm]

            loss = 0.
            for k in range(0,obsv_samples.shape[0], batch_size):
                loss += model.update(session, obsv_samples[k:k+batch_size],
                                     action_samples[k:k+batch_size])

            new_exp = model.test_run(session, env, max_steps)

            #Data Aggregation Steps. Supervision signal comes from expert policy.
            new_exp_len = new_exp['observations'].shape[0]
            expert_expected_actions = []
            for k in range(0, new_exp_len, batch_size) :
                expert_expected_actions.append(expert_policy_fn(new_exp['observations'][k:k+batch_size]))

            # add new experience into original one. (No eviction)
            data['observations'] = np.concatenate((data['observations'], new_exp['observations']),
                                                  axis=0)
            data['actions'] = np.concatenate([data['actions']] + expert_expected_actions,
                                             axis=0)
            tqdm.write("Epoch %3d Loss %f Reward %f" %(epoch, loss/num_samples, new_exp['reward']))

        env = wrappers.Monitor(env, results_dir, force=True)

        results = []
        for _ in tqdm(range(num_rollouts)):
            results.append(model.test_run(session, env, max_steps )['reward'])
        logger.info("Reward mean & std of Cloned policy with DAGGER: %f(%f)"%(np.mean(results), np.std(results)))
    return np.mean(data['returns']), np.std(data['returns']), np.mean(results), np.std(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rollouts', type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=.001)
    parser.add_argument("--restore", type=bool, default=False)
    args = parser.parse_args()

    log_file = os.path.join(os.getcwd(), 'results', 'train_out.log')
    logger = config_logging(log_file)

    env_models = [('Ant-v1','experts/Ant-v1.pkl'),
                  ('HalfCheetah-v1','experts/HalfCheetah-v1.pkl'),
                  ('Hopper-v1','experts/Hopper-v1.pkl'),
                  ('Humanoid-v1','experts/Humanoid-v1.pkl'),
                  ('Reacher-v1','experts/Reacher-v1.pkl'),
                  ('Walker2d-v1','experts/Walker2d-v1.pkl'),]

    results = []
    for env_name, expert_data in env_models :
        bc_results_dir = os.path.join(os.getcwd(), 'results', env_name, 'bc')
        if not os.path.exists(bc_results_dir):
            os.makedirs(bc_results_dir)
        ex_mean, ex_std, bc_mean,bc_std = bc(expert_data, args.num_rollouts,
            args.num_epochs, args.optimizer, args.learning_rate, env_name, args.batch_size, args.restore, bc_results_dir)

        da_results_dir = os.path.join(os.getcwd(), 'results', env_name, 'da')
        if not os.path.exists(da_results_dir):
            os.makedirs(da_results_dir)
        _,_, da_mean,da_std = dagger(expert_data, env_name, args.optimizer, args.learning_rate, args.restore, da_results_dir,
                                num_epochs=40,
                                batch_size=args.batch_size)
        results.append((env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std))

    for env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std in results :
        logger.info('Env: %s, Expert: %f(%f), Behavior Cloning: %f(%f), Dagger: %f(%f)'%
              (env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std))
        