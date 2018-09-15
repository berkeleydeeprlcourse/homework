import argparse
import pickle
from collections import namedtuple
from time import sleep

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import multiprocessing
import numpy as np

import gym
import load_policy
import tf_util

import keras


#############
# Stats
#############

class DAggerStats(object):
    def __init__(self):
        self.DAggerData = namedtuple('DAggerData', 'num_datapts num_iterations reward_mean reward_std_dev')
        self.dataset = []

    def log(self, tup):
        self.dataset.append(tup._asdict())

    def print(self):
        for datapt in self.dataset:
            print(datapt)

    def save(self, fname):
        import csv
        keys = self.dataset[0].keys()
        with open(fname, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.dataset)


###############
# Expert policy
###############

class ExpertPolicy(object):
    def __init__(self, expert_policy_file):
        self.expert_policy_file = expert_policy_file
        self.g = tf.Graph()

    def run_expert(self, observations):
        actions = []
        with tf.Session(graph=self.g):
            tf_util.initialize()
            policy_fn = load_policy.load_policy(self.expert_policy_file)
            for observation in observations:
                act = policy_fn(observation[None, :])
                actions.append(act)
        return np.stack(actions, axis=0)


###############
# DAgger policy
###############

class DAggerPolicy(object):
    def __init__(self, envname, hidden_units, in_size):
        self.in_size = in_size
        self.hidden_units = hidden_units
        self.envname = envname
        self.save_name = 'dagger_data/{}/model.ckpt'.format(envname)
        self.model = self.create_model()

    def create_model(self):
        model = keras.models.Sequential()
        for i, hu in enumerate(self.hidden_units):
            if i == 0:
                model.add(keras.layers.Dense(hu, input_dim=self.in_size, activation='relu', use_bias=True))
            elif i == len(self.hidden_units) - 1:
                model.add(keras.layers.Dense(hu, input_dim=self.in_size, use_bias=True))
            else:
                model.add(keras.layers.Dense(hu, activation='relu'))
        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        return model

    def train_model(self, x_train, y_train, num_epochs, batch_size=32):
        self.model.fit(x_train, np.squeeze(y_train), epochs=num_epochs, batch_size=batch_size, verbose=1,
                       validation_split=0)

    def eval_policy(self, num_rollouts=None, render=False):
        env = gym.make(self.envname)
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = self.model.predict(np.expand_dims(obs, axis=0))
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render and i == 0:
                    sleep(2)
                    env.render()
                    sleep(2)
                    input("Press Enter to continue...")
                if steps >= env.spec.timestep_limit:
                    break
            returns.append(totalr)
        return np.stack(observations), actions, returns


def load_dataset(task):
    data = pickle.load(open('expert_data/{}.pkl'.format(task), 'rb'))
    obvs, acts = data['observations'], data['actions']
    x_train, x_test, y_train, y_test = train_test_split(obvs, acts, test_size=0 / 100.)
    return x_train, y_train


def main(envname: str, render: bool = False, num_rollouts: int = 20, num_epochs: int = 10,
         batch_size: int = 128, num_iterations: int = 100):
    stats = DAggerStats()
    obvs, acts = load_dataset(envname)
    daggerPolicy = DAggerPolicy(envname, [80, 60, 40, np.prod(acts.shape[1:])], obvs.shape[1])
    expert = ExpertPolicy("experts/{}.pkl".format(envname))
    for i in range(num_iterations):
        obvs, acts = shuffle(obvs, acts)
        daggerPolicy.train_model(obvs, acts, num_epochs=num_epochs, batch_size=batch_size)
        print("trained")

        _, _, returns = daggerPolicy.eval_policy(num_rollouts=25, render=False)
        stats.log(stats.DAggerData(obvs.shape[0], i, np.mean(returns), np.std(returns)))

        # run model over samples and log mean + std_dev of returns (to collect observations)
        observations, _, _ = daggerPolicy.eval_policy(num_rollouts=num_rollouts, render=render)
        print("rolled out obvs")

        # run expert over observations and add to dataset
        actions = expert.run_expert(observations)
        print("ran expert")

        obvs = np.concatenate((obvs, observations), axis=0)
        acts = np.concatenate((acts, actions), axis=0)

        stats.print()
        stats.save("dagger_data/{}.pkl".format(envname))

    stats.save("dagger_data/{}.pkl".format(envname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=1, help='Number of DAgger iteration rollouts')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_iterations', type=int, default=256)
    args = parser.parse_args()
    main(**vars(args))
