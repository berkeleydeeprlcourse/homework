from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pickle_util import load_obj, save_obj

# hyperparameters
GRADIENT_DESCENT_STEP_SIZE = 0.000005
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 1000
TRAINING_STEPS = 10000

def feedforward_nn(x, observation_size, output_size, hidden_size=HIDDEN_LAYER_SIZE):
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

def import_observations():
  expert_data = load_obj('expert_data.pkl')
  return expert_data['observations']


def import_actions():
  expert_data = load_obj('expert_data.pkl')
  actions = expert_data['actions']
  # we have to flatten the array, because they are column vectors for some reason
  return np.array([v.flatten() for v in actions])

def main(_):
  # Import data
  observations = import_observations()
  actions = import_actions()

  observation_size = len(observations[0])
  action_size = len(actions[0])

  # Create the model
  x = tf.placeholder(tf.float32, [None, observation_size])
  y = feedforward_nn(x, observation_size, action_size)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, action_size])

  l2_loss = tf.nn.l2_loss(y_ - y)
  train_step = tf.train.GradientDescentOptimizer(GRADIENT_DESCENT_STEP_SIZE).minimize(l2_loss)

  # Train
  train_accuracies = np.zeros([TRAINING_STEPS])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
      indexes = np.random.choice(observations.shape[0], BATCH_SIZE, replace=False)
      batch_xs = observations[indexes, :]
      batch_ys = actions[indexes, :]
      train_accuracy = l2_loss.eval(feed_dict={x: batch_xs, y_: batch_ys})
      train_accuracies[i] = train_accuracy
      if i % 100 == 0:
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

  save_obj(train_accuracies, 'training_accuracies.pkl')
  plt.plot(train_accuracies)
  plt.ylabel('training accuracy')
  plt.xlabel('training steps')
  plt.show()

if __name__ == '__main__':
  tf.app.run(main=main)
  