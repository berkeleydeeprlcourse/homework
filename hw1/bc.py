from __future__ import print_function
import os
import sys
import logging
import argparse
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers

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

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def create_model(session, num_observations, num_actions, logger, optimizer, learning_rate, restore, checkpoint_dir):
    model = Model(num_observations, num_actions, checkpoint_dir, optimizer, learning_rate)

    if restore:
        model.load(session)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model

def train(expert_data, log_dir, num_epochs, optimizer, learning_rate, env_name, batch_size, restore, val, checkpoint_dir):
    logger = config_logging(log_dir + "bc_{}_train_out.log".format(env_name))

    # split expert data into train and validation sets
    data = Data(expert_data, train_ratio=0.9, val_ratio=0.05)
    steps_per_print = 100

    num_train = len(data.train['observations'])
    num_batches_per_epoch = int((num_train - 1) / batch_size) + 1

    # intialize avg loss and minimum validation loss
    avg_loss, min_val_loss = 0, sys.maxsize

    with tf.Session() as session:
        model = create_model(session, data.num_observations, data.num_actions, logger, optimizer, learning_rate, restore, checkpoint_dir)

        batches = data.batch_iter(data.train, batch_size, num_epochs)
        for i, (batch_x, batch_y) in enumerate(batches):
            pred, loss = model.step(session, batch_x, batch_y)

            avg_loss += loss
            num_epoch = i / num_batches_per_epoch

            if i % steps_per_print == 0:
                logger.debug("Epoch %04d step %08d loss %04f" % (num_epoch, i, loss))

            if i > 0 and (i+1) % num_batches_per_epoch == 0:
                avg_loss /= num_batches_per_epoch
                logger.debug("###################################")
                logger.info("Finished epoch %d, average training loss = %f" % (num_epoch, avg_loss))
                if val:
                    min_val_loss = validate(
                        model, logger, session, data, num_epochs, batch_size, min_val_loss, checkpoint_dir)
                logger.debug("###################################")
                avg_loss = 0

def validate(model, logger, session, data, num_epoch, batch_size, min_loss, checkpoint_dir):
    batches = data.batch_iter(data.val, batch_size, 1)
    avg_loss = []
    for i, (batch_x, batch_y) in enumerate(batches):
        pred, loss = model.step(session, batch_x, batch_y, is_train=False)
        avg_loss.append(loss)
    new_loss = sum(avg_loss) / len(avg_loss)
    logger.info("Finished epoch %d, average validation loss = %f" % (num_epoch, new_loss))
    if new_loss < min_loss:  # Only save model if val loss dropped
        model.save(session)
        logger.info("Model saved!")
        min_loss = new_loss
    return min_loss

def test(expert_data, log_dir, optimizer, learning_rate, env_name, num_rollouts, checkpoint_dir):
    logger = config_logging(log_dir + "bc_{}_test_out.log".format(env_name))

    # create to get num_observations and num_actions for model
    data = Data(expert_data, train_ratio=0.9, val_ratio=0.05)

    with tf.Session() as session:
        model = create_model(session, data.num_observations, data.num_actions, logger, optimizer, learning_rate, True, checkpoint_dir)
        env = gym.make(env_name)
        max_steps = env.spec.timestep_limit

        returns, observations, actions = [], [], []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            total_reward = 0.
            steps = 0
            while not done:
                action = model.step(session, obs[None, :], is_train=False)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                total_reward += r
                steps += 1
                env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(total_reward)

        print('returns', returns)
        print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_data", type=str, default="data/Hopper-v1_data_1000_rollouts_500_timesteps.pkl")
    parser.add_argument('--env_name', type=str, default="Hopper-v1")
    parser.add_argument('--num_rollouts', type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--ckpt_dir", type=str, default="results/")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--restore", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--val", type=bool, default=True)
    parser.add_argument("--overfit", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=True)
    args = parser.parse_args()
    
    fullcheckpoint_dir = args.ckpt_dir + "bc/{}/".format(args.env_name)

    if args.train:
        train(args.expert_data, args.log_dir, args.num_epochs, args.optimizer, args.learning_rate, args.env_name, args.batch_size, args.restore, args.val, fullcheckpoint_dir)
    elif args.test:
        test(args.expert_data, args.log_dir, args.optimizer, args.learning_rate, args.env_name, args.num_rollouts, fullcheckpoint_dir)

        