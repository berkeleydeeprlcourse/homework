import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import itertools

class Model:
    def __init__(self, obs_samples, num_observations, num_actions, checkpoint_dir, logger, optimizer, learning_rate):
        self.logger = logger

        self.obs_mean = obs_samples.mean(axis=0)
        self.obs_std = obs_samples.std(axis=0)

        self.num_observations = num_observations
        self.num_actions = num_actions
        
        self.obs = tf.placeholder(tf.float32, [None, num_observations])
        self.actions = tf.placeholder(tf.float32, [None, num_actions])
        self.init_global_step()
        self.pred, self.parameters = self.build_model()

        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer(optimizer, learning_rate)

        self.checkpoint_dir = checkpoint_dir
        self.saver = tf.train.Saver(var_list=tf.global_variables())

    def save(self, sess):
        self.logger.info("Saving model...")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(sess, self.checkpoint_dir + 'model', global_step=self.global_step_tensor)
        self.logger.info("Model saved")

    def load(self, session):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.logger.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(session, latest_checkpoint)
        self.logger.info("Model loaded")

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

    def fc_layer(self, input_tensor, input_dim, output_dim, layer_name, parameters, activation='relu'):
        with tf.name_scope(layer_name):
            weights = tf.get_variable('weights_' + layer_name, [input_dim, output_dim],
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.variable_summaries(weights)

            biases = tf.get_variable('biases_' + layer_name, [output_dim],
                initializer=tf.constant_initializer(0), dtype=tf.float32)
            self.variable_summaries(biases)

            z_fc1 = tf.add(tf.matmul(input_tensor, weights), biases)
            tf.summary.histogram('z_' + layer_name, z_fc1)

            if activation == 'relu':
                a_fc1 = tf.nn.relu(z_fc1)
                tf.summary.histogram('a_' + layer_name, a_fc1)
            else:
                a_fc1 = z_fc1

            parameters += [weights, biases]
            return a_fc1

    def build_model(self):
        parameters = []

        normalized = (self.obs - self.obs_mean) / self.obs_std

        net = slim.fully_connected(normalized, 50, scope='fc1', activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, 50, scope='fc2', activation_fn=tf.nn.relu)
        policy = slim.fully_connected(net, self.num_actions, activation_fn=None, scope='policy')



        # a_fc1 = self.fc_layer(normalized, self.num_observations, 128, 'fc1', parameters)
        # a_fc2 = self.fc_layer(a_fc1, 128, 128, 'fc2', parameters)
        # a_fc3 = self.fc_layer(a_fc2, 128, 128, 'fc3', parameters)
        # z_fc4 = self.fc_layer(a_fc3, 128, self.num_actions, 'fc4', parameters, activation = None)
        return policy, parameters

    def get_optimizer(self, optimizer, learning_rate):
        self.logger.info("Using %s optimizer" % optimizer)
        if optimizer == "adam":
            return tf.train.AdamOptimizer(learning_rate).minimize(self.loss,
                global_step=self.global_step_tensor)
        elif optimizer == "adagrad":
            return tf.train.AdagradOptimizer(learning_rate).minimize(self.loss,
                global_step=self.global_step_tensor)
        else:
            return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss,
                global_step=self.global_step_tensor)

    def get_loss(self):
        loss = tf.reduce_mean(tf.reduce_sum((self.pred - self.actions)**2, axis=1))
        # loss = tf.reduce_mean(tf.pow(self.pred - self.actions, 2)) / 2
        return loss

    def predict(self, sess, batch_x):
        return sess.run(self.pred,
                        feed_dict={self.obs:batch_x})

    def update(self, sess, batch_x, batch_y):
        loss, _ = sess.run([self.loss, self.optimizer],
                          feed_dict={self.obs: batch_x,
                                     self.actions: batch_y})
        return loss

    def test_run(self, sess, env, max_steps):
        obvs = []
        actions = []
        reward = 0.

        obv = env.reset()
        for steps in itertools.count() :
            obvs.append(obv)
            actions.append(self.predict(sess, np.expand_dims(obv,axis=0))[0])
            obv, r, done, _ = env.step(actions[-1])
            reward += r
            if steps >= max_steps or done:
                break

        experience = {'observations': np.stack(obvs,axis=0),
                      'actions': np.squeeze(np.stack(actions,axis=0)),
                      'reward':reward}
        return experience