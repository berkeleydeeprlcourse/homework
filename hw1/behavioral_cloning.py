# example usage
# python behavioral_cloning.py Hopper-v1 --expert_data_filename expert_data/expert_data_hopper.pkl --model_filepath /tmp/model
# python behavioral_cloning.py Ant-v1 --expert_data_filename expert_data/expert_data_ant.pkl --model_filepath /tmp/model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pickle_util import load_obj, save_obj

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

def main(expert_data_filename,
         envname,
         model_filepath,
         hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE,
         batch_size=DEFAULT_BATCH_SIZE,
         training_steps=DEFAULT_TRAINING_STEPS,
         gradient_descent_step_size=DEFAULT_GRADIENT_DESCENT_STEP_SIZE):
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

  # Train
  train_accuracies = np.zeros([training_steps])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(training_steps):
      indexes = np.random.choice(observations.shape[0], batch_size, replace=False)
      batch_xs = observations[indexes, :]
      batch_ys = actions[indexes, :]
      train_accuracy = l2_loss.eval(feed_dict={x: batch_xs, y_: batch_ys})
      train_accuracies[i] = train_accuracy
      if i % 100 == 0:
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

    # save the trained model parameters

    saver = tf.train.Saver()
    saver.save(sess, model_filepath)

  save_obj(train_accuracies, 'training_accuracies/training_accuracies_{}.pkl'.format(envname))
  plot_and_save_figure(train_accuracies, envname)

  return train_accuracies


def plot_and_save_figure(train_accuracies, envname):
  plt.plot(train_accuracies)
  plt.ylabel('training accuracy')
  plt.xlabel('training steps')
  plt.savefig('training_accuracies/training_accuracies_' + envname)
  plt.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('envname', type=str)
  parser.add_argument('--expert_data_filename', type=str)
  parser.add_argument('--model_filepath', type=str)
  parser.add_argument('--hidden_layer_size', type=int, default=DEFAULT_HIDDEN_LAYER_SIZE)
  parser.add_argument('--training_steps', type=int, default=DEFAULT_TRAINING_STEPS)
  args = parser.parse_args()

  main(args.expert_data_filename,
       args.envname,
       args.model_filepath,
       training_steps=args.training_steps,
       hidden_layer_size=args.hidden_layer_size)

