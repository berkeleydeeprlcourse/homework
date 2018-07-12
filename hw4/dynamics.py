import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_actions, self.std_actions = normalization
        self.batch_size = batch_size
        self.iterations = iterations

        self.sess = sess
        # input placeholder, state/action pairs
        self.state_act_placeholder = tf.placeholder(shape = [None, env.observation_space.shape[0] + env.action_space.shape[0]], 
                                            name = 'input_state_act', dtype = tf.float32)
        # labels                                                
        self.deltas_placeholder = tf.placeholder(shape = [None, env.observation_space.shape[0]], name = 'deltas', dtype = tf.float32)
        
        # build MLP
        self.deltas_predict = build_mlp(self.state_act_placeholder, env.observation_space.shape[0], 
                                            scope = 'model', n_layers = n_layers, size = size,
                                            activation = activation, output_activation = output_activation)
        
        # MSE between deltas predicted and actual
        self.loss = tf.reduce_mean(tf.square(self.deltas_predict - self.deltas_placeholder))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        """ Note: Be careful about normalization """

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        # flatten
        observations = np.concatenate([item['observations'] for item in data])
        actions = np.concatenate([item['actions'] for item in data])
        next_observations = np.concatenate([item['next_observations'] for item in data])        
        
        # normalize states and actions
        obs_norm = (observations - self.mean_obs) / (self.std_obs + 1e-7)
        acts_norm = (actions - self.mean_actions) / (self.std_actions + 1e-7)

        # normalize the state differences
        deltas_obs_norm = ((next_observations - observations) - self.mean_deltas) / (self.std_deltas + 1e-7)

        obs_act_norm = np.concatenate((obs_norm, acts_norm), axis = 1)
        
        train_indices = np.arange(observations.shape[0])
        
        for i in range(self.iterations):
            np.random.shuffle(train_indices)

            for j in range((observations.shape[0] // self.batch_size) + 1):
                start_index = j * self.batch_size
                indices_shuffled = train_indices[start_index:start_index + self.batch_size]
                
                input_batch = obs_act_norm[indices_shuffled, :]
                label_batch = deltas_obs_norm[indices_shuffled, :]
                
                self.sess.run([self.train_op], feed_dict = {self.state_act_placeholder: input_batch, self.deltas_placeholder: label_batch})
            


    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        # normalize the states and actions
        obs_norm = (states - self.mean_obs) / (self.std_obs + 1e-7)
        act_norm = (actions - self.mean_actions) / (self.std_actions + 1e-7)

        # concatenate normalized states and actions
        obs_act_norm = np.concatenate((obs_norm, act_norm), axis=1 )
        
        # predict the deltas between states and next states
        deltas = self.sess.run(self.deltas_predict, feed_dict = {self.state_act_placeholder: obs_act_norm})
        
        # calculate the next states using the predicted delta values and denormalize
        return deltas * self.std_deltas + self.mean_deltas + states
