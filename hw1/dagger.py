# example usage
# generate expert data
# python run_expert.py experts/Ant-v1.pkl Ant-v1 --num_rollouts 20 --expert_data_filename expert_data/expert_data_Ant-v1.pkl
# run dagger
# python dagger.py Ant-v1 --expert_data_filename expert_data/expert_data_Ant-v1.pkl --expert_policy_file experts/Ant-v1.pkl

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gym

from pickle_util import load_obj
import tf_util
import load_policy

# hyperparameters
DEFAULT_GRADIENT_DESCENT_STEP_SIZE = 0.000005
DEFAULT_HIDDEN_LAYER_SIZE = 64
DEFAULT_BATCH_SIZE = 1000
DEFAULT_TRAINING_STEPS = 10000

def feedforward_nn(x, observation_size, output_size, hidden_size):
  # x is of size OBSERVATION_SIZE

  # fully connected input layer
  with tf.name_scope('fc1'):
    W_fc1 = tf.Variable(tf.random_normal([observation_size, hidden_size], stddev=0.1))
    b_fc1 = tf.Variable(tf.zeros([hidden_size]))
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

  # fully connected output layer
  with tf.name_scope('fc2'):
    W_fc2 = tf.Variable(tf.random_normal([hidden_size, output_size], stddev=0.1))
    b_fc2 = tf.Variable(tf.zeros([output_size]))
    y = tf.matmul(h_fc1, W_fc2) + b_fc2

  return y

def import_observations(expert_data_filename):
  expert_data = load_obj(expert_data_filename)
  return expert_data['observations']


def import_actions(expert_data_filename):
  expert_data = load_obj(expert_data_filename)
  actions = expert_data['actions']
  # we have to flatten the array, because they are column vectors for some reason
  return np.array([v.flatten() for v in actions])

def training_loop(sess, observations, actions, batch_size, training_steps, x, y_, loss, train_step,):
    train_accuracies = np.zeros([training_steps])

    sess.run(tf.global_variables_initializer())
    for i in range(training_steps):
        indexes = np.random.choice(observations.shape[0], batch_size, replace=False)
        batch_xs = observations[indexes, :]
        batch_ys = actions[indexes, :]
        train_accuracy = loss.eval(feed_dict={x: batch_xs, y_: batch_ys})
        train_accuracies[i] = train_accuracy
        if i % 100 == 0:
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    return train_accuracies

def run_policy_loop(sess,
                    envname,
                    num_rollouts,
                    y,
                    x,
                    max_timesteps=None,
                    render=False):
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
            # need to turn this into a vector of inputs
            action = sess.run(y, {x: obs[None,:]})
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
    return (np.array(observations), np.array(actions), returns)

def get_expert_actions(observations, expert_policy_file):
    policy_fn = load_policy.load_policy(expert_policy_file)

    tf_util.initialize()

    actions = policy_fn(observations)
    return actions

def dagger(expert_data_filename,
           envname,
           expert_policy_file,
           hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
           batch_size=DEFAULT_BATCH_SIZE,
           training_steps=DEFAULT_TRAINING_STEPS,
           gradient_descent_step_size=DEFAULT_GRADIENT_DESCENT_STEP_SIZE,
           num_rollouts=20,
           dagger_iterations=5):
    # Import data
    observations = import_observations(expert_data_filename)
    actions = import_actions(expert_data_filename)

    observation_size = len(observations[0])
    action_size = len(actions[0])

    # Create the model
    x = tf.placeholder(tf.float32, [None, observation_size])
    y = feedforward_nn(x, observation_size, action_size, hidden_layer_size)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, action_size])

    l2_loss = tf.nn.l2_loss(y_ - y)
    train_step = tf.train.GradientDescentOptimizer(gradient_descent_step_size).minimize(l2_loss)

    train_accuracies = np.array([])
    mean_rewards = []
    with tf.Session() as sess:
        for i in range(dagger_iterations):
            print('running dagger iteration {}'.format(i))
            # train model on observations
            iteration_accuracies = training_loop(sess, observations, actions, batch_size, training_steps, x, y_, l2_loss, train_step)
            train_accuracies = np.append(train_accuracies, iteration_accuracies)
            # generate states with policy
            new_observations, new_actions, rewards = run_policy_loop(sess, envname, num_rollouts, y, x)
            mean_rewards.append(np.mean(rewards))
            # ask for expert actions
            new_actions = get_expert_actions(new_observations, expert_policy_file)
            observations = np.append(observations, new_observations, axis=0)
            actions = np.append(actions, new_actions, axis=0)

    plot_and_save_training_accuracies(train_accuracies, envname)
    plot_and_save_dagger_reward(mean_rewards, envname)
    return train_accuracies

def plot_and_save_dagger_reward(mean_rewards, envname):
  plt.plot(mean_rewards)
  plt.ylabel('mean dagger reward accuracy')
  plt.xlabel('dagger iterations')
  plt.savefig('dagger_mean_reward_' + envname)
  plt.close()


def plot_and_save_training_accuracies(train_accuracies, envname):
  plt.plot(train_accuracies)
  plt.ylabel('training accuracy')
  plt.xlabel('training steps')
  plt.savefig('training_accuracies/dagger_training_accuracies_' + envname)
  plt.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('envname', type=str)
  parser.add_argument('--expert_data_filename', type=str)
  parser.add_argument('--expert_policy_file', type=str)
  parser.add_argument('--hidden_layer_size', type=int, default=DEFAULT_HIDDEN_LAYER_SIZE)
  parser.add_argument('--training_steps', type=int, default=DEFAULT_TRAINING_STEPS)
  parser.add_argument('--num_rollouts', type=int, default=20)
  args = parser.parse_args()

  dagger(args.expert_data_filename,
         args.envname,
         args.expert_policy_file,
         hidden_layer_size=args.hidden_layer_size,
         training_steps=args.training_steps,
         num_rollouts=args.num_rollouts)
