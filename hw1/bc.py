from __future__ import print_function
import os
import sys
import logging
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import load_policy
import pickle
from sklearn.model_selection import train_test_split

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
        logger.info("Created model with fresh parameters")
        session.run(tf.global_variables_initializer())

    return model

def gather_expert_experience(num_rollouts, env, policy_fn, max_steps):
    with tf.Session():
        returns = []
        observations = []
        actions = []
        for _ in tqdm(range(num_rollouts)):
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


def bc(expert_data_file, expert_policy_file, env_name, restore, results_dir,
            num_rollouts, max_timesteps=None, optimizer='adam', num_epochs=100, learning_rate=.001, batch_size=32, keep_prob=1):
    tf.reset_default_graph()
    
    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    # data = Data(expert_data_file, train_ratio=0.9, val_ratio=0.05)

    with open(expert_data_file, 'rb') as f:
        data = pickle.loads(f.read())

    obs = np.stack(data['observations'], axis=0)
    actions = np.squeeze(np.stack(data['actions'], axis=0))

    x_train, x_test, y_train, y_test = train_test_split(obs, actions, test_size=0.2)

    num_samples = len(x_train)

    min_val_loss = sys.maxsize

    with tf.Session() as session:
        model = create_model(session, x_train, x_train.shape[1], y_train.shape[1], logger, optimizer, learning_rate, restore, results_dir)

        file_writer = tf.summary.FileWriter(results_dir, session.graph)

        for epoch in tqdm(range(num_epochs)):
            perm = np.random.permutation(x_train.shape[0])

            obs_samples = x_train[perm]
            action_samples = y_train[perm]

            loss = 0.
            for k in range(0,obs_samples.shape[0], batch_size):
                batch_loss, training_scalar = model.update(session, obs_samples[k:k+batch_size],
                                     action_samples[k:k+batch_size],
                                     keep_prob)
                loss += batch_loss

            file_writer.add_summary(training_scalar, epoch)

            min_val_loss, validation_scalar = validate(model, logger, session, x_test, y_test, epoch, batch_size, min_val_loss, results_dir)
            file_writer.add_summary(validation_scalar, epoch)

            new_exp = model.test_run(session, env, max_steps )
            tqdm.write("Epoch %3d Loss %f Reward %f" %(epoch, loss/num_samples, new_exp['reward']))

        env = wrappers.Monitor(env, results_dir, force=True)

        results = []
        for _ in tqdm(range(10)):
            results.append(model.test_run(session, env, max_steps )['reward'])
        logger.info("Reward mean and std dev with behavior cloning: %f(%f)"%(np.mean(results), np.std(results)))
    return data['mean_return'], data['std_return'], np.mean(results), np.std(results)

def validate(model, logger, session, x_test, y_test, num_epoch, batch_size, min_loss, checkpoint_dir):
    avg_loss = []

    # for k in range(0, x_test.shape[0], batch_size):
    loss, validation_scalar = model.validate(session, x_test, y_test)
    avg_loss.append(loss)

    new_loss = sum(avg_loss) / len(avg_loss)
    logger.info("Finished epoch %d, average validation loss = %f" % (num_epoch, new_loss))

    if new_loss < min_loss:  # Only save model if val loss dropped
        model.save(session)
        min_loss = new_loss
    return min_loss, validation_scalar

def dagger(expert_data_file, expert_policy_file, env_name, restore, results_dir,
            num_rollouts, max_timesteps=None, optimizer='adam', num_epochs=40, learning_rate=.001, batch_size=32, keep_prob=1):
    tf.reset_default_graph()

    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    expert_policy_fn = load_policy.load_policy(expert_policy_file)

    # data = Data(expert_data_file, train_ratio=0.9, val_ratio=0.05)

    with open(expert_data_file, 'rb') as f:
        data = pickle.loads(f.read())

    obs = np.stack(data['observations'], axis=0)
    actions = np.squeeze(np.stack(data['actions'], axis=0))

    x_train, x_test, y_train, y_test = train_test_split(obs, actions, test_size=0.2)

    min_val_loss = sys.maxsize

    with tf.Session() as session:
        model = create_model(session, x_train, x_train.shape[1], y_train.shape[1], logger, optimizer, learning_rate, restore, results_dir)

        file_writer = tf.summary.FileWriter(results_dir, session.graph)

        for epoch in tqdm(range(num_epochs)):
            num_samples = x_train.shape[0]
            perm = np.random.permutation(num_samples)

            obsv_samples = x_train[perm]
            action_samples = y_train[perm]

            obsv_samples = np.stack(obsv_samples, axis=0)
            action_samples = np.squeeze(np.stack(action_samples, axis=0))


            loss = 0.
            for k in range(0,obsv_samples.shape[0], batch_size):
                batch_loss, training_scalar = model.update(session, obsv_samples[k:k+batch_size],
                                     action_samples[k:k+batch_size],
                                     keep_prob)
                loss += batch_loss
            
            file_writer.add_summary(training_scalar, epoch)                        

            min_val_loss, validation_scalar = validate(model, logger, session, x_test, y_test, epoch, batch_size, min_val_loss, results_dir)
            file_writer.add_summary(validation_scalar, epoch)

            new_exp = model.test_run(session, env, max_steps)

            #Data Aggregation Steps. Supervision signal comes from expert policy.
            new_exp_len = new_exp['observations'].shape[0]
            expert_expected_actions = []
            for k in range(0, new_exp_len, batch_size) :
                expert_expected_actions.append(expert_policy_fn(new_exp['observations'][k:k+batch_size]))

            # add new experience into original one. (No eviction)
            x_train = np.concatenate((x_train, new_exp['observations']),
                                                  axis=0)
            y_train = np.concatenate([y_train] + expert_expected_actions,
                                             axis=0)
            tqdm.write("Epoch %3d Loss %f Reward %f" %(epoch, loss/num_samples, new_exp['reward']))

        env = wrappers.Monitor(env, results_dir, force=True)

        results = []
        for _ in tqdm(range(10)):
            results.append(model.test_run(session, env, max_steps )['reward'])
        logger.info("Reward mean and std dev with DAgger: %f(%f)"%(np.mean(results), np.std(results)))
    return data['mean_return'], data['std_return'], np.mean(results), np.std(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", type=bool, default=False)
    args = parser.parse_args()

    log_file = os.path.join(os.getcwd(), 'results', 'train_out.log')
    logger = config_logging(log_file)

    env_models = [('Ant-v1', 'data/Ant-v1_data_250_rollouts.pkl', 'experts/Ant-v1.pkl', 250),
                  ('HalfCheetah-v1', 'data/HalfCheetah-v1_data_10_rollouts.pkl', 'experts/HalfCheetah-v1.pkl', 10),
                  ('Hopper-v1', 'data/Hopper-v1_data_10_rollouts.pkl', 'experts/Hopper-v1.pkl',  10),
                  ('Humanoid-v1', 'data/Humanoid-v1_data_250_rollouts.pkl', 'experts/Humanoid-v1.pkl', 250),
                  ('Reacher-v1', 'data/Reacher-v1_data_250_rollouts.pkl', 'experts/Reacher-v1.pkl', 250),
                  ('Walker2d-v1', 'data/Walker2d-v1_data_10_rollouts.pkl','experts/Walker2d-v1.pkl', 10)
                  ]

    results = []
    for env_name, rollout_data, expert_policy_file, num_rollouts in env_models :
        bc_results_dir = os.path.join(os.getcwd(), 'results', env_name, 'bc')
        if not os.path.exists(bc_results_dir):
            os.makedirs(bc_results_dir)
        ex_mean, ex_std, bc_mean,bc_std = bc(rollout_data, expert_policy_file, env_name, args.restore, bc_results_dir, num_rollouts)

        da_results_dir = os.path.join(os.getcwd(), 'results', env_name, 'da')
        if not os.path.exists(da_results_dir):
            os.makedirs(da_results_dir)
        _,_, da_mean,da_std = dagger(rollout_data, expert_policy_file, env_name, args.restore, da_results_dir, num_rollouts)
        results.append((env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std))

    for env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std in results :
        logger.info('Env: %s, Expert: %f(%f), Behavior Cloning: %f(%f), Dagger: %f(%f)'%
              (env_name, ex_mean, ex_std, bc_mean, bc_std, da_mean, da_std))
        