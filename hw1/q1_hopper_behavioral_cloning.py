from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import tensorflow as tf

HIDDEN_SIZE = 64
OBSERVATION_SIZE = 11
ACTION_SIZE = 3

def feedforward_nn(x):
  # x is of size OBSERVATION_SIZE

  # fully connected input layer
  with tf.name_scope('fc1'):
    W_fc1 = tf.Variable(tf.random_normal([OBSERVATION_SIZE, HIDDEN_SIZE], stddev=0.1))
    b_fc1 = tf.Variable(tf.zeros([HIDDEN_SIZE]))
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
  
  # fully connected output layer
  with tf.name_scope('fc2'):
    W_fc2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, ACTION_SIZE], stddev=0.1))
    b_fc2 = tf.Variable(tf.zeros([ACTION_SIZE]))
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

  # Create the model
  x = tf.placeholder(tf.float32, [None, OBSERVATION_SIZE])
  y = feedforward_nn(x)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, ACTION_SIZE])

  l2_loss = tf.nn.l2_loss(y_ - y)
  train_step = tf.train.GradientDescentOptimizer(0.000005).minimize(l2_loss)

  # Train
  batch_size = 1000
  training_steps = 10000
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

  import matplotlib.pyplot as plt
  plt.plot(train_accuracies)
  plt.ylabel('training accuracy')
  plt.xlabel('training steps')
  plt.show()

def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
  tf.app.run(main=main)
  