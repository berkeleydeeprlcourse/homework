import numpy as np
import tensorflow as tf

from logger import logger


############
### Data ###
############

class Dataset(object):

    def __init__(self):
        self._states = []
        self._actions = []
        self._next_states = []
        self._rewards = []
        self._dones = []

    @property
    def is_empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._states)

    ##################
    ### Statistics ###
    ##################

    @property
    def state_mean(self):
        return np.mean(self._states, axis=0)

    @property
    def state_std(self):
        return np.std(self._states, axis=0)

    @property
    def action_mean(self):
        return np.mean(self._actions, axis=0)

    @property
    def action_std(self):
        return np.std(self._actions, axis=0)

    @property
    def delta_state_mean(self):
        return np.mean(np.array(self._next_states) - np.array(self._states), axis=0)

    @property
    def delta_state_std(self):
        return np.std(np.array(self._next_states) - np.array(self._states), axis=0)

    ###################
    ### Adding data ###
    ###################

    def add(self, state, action, next_state, reward, done):
        """
        Add (s, a, r, s') to this dataset
        """
        if not self.is_empty:
            # ensure the state, action, next_state are of the same dimension
            assert len(self._states[-1]) == len(np.ravel(state))
            assert len(self._actions[-1]) == len(np.ravel(action))
            assert len(self._next_states[-1]) == len(np.ravel(next_state))

        self._states.append(np.ravel(state))
        self._actions.append(np.ravel(action))
        self._next_states.append(np.ravel(next_state))
        self._rewards.append(reward)
        self._dones.append(done)

    def append(self, other_dataset):
        """
        Append other_dataset to this dataset
        """
        if not self.is_empty and not other_dataset.is_empty:
            # ensure the state, action, next_state are of the same dimension
            assert len(self._states[-1]) == len(other_dataset._states[-1])
            assert len(self._actions[-1]) == len(other_dataset._actions[-1])
            assert len(self._next_states[-1]) == len(other_dataset._next_states[-1])

        self._states += other_dataset._states
        self._actions += other_dataset._actions
        self._next_states += other_dataset._next_states
        self._rewards += other_dataset._rewards
        self._dones += other_dataset._dones

    ############################
    ### Iterate through data ###
    ############################

    def rollout_iterator(self):
        """
        Iterate through all the rollouts in the dataset sequentially
        """
        end_indices = np.nonzero(self._dones)[0] + 1

        states = np.asarray(self._states)
        actions = np.asarray(self._actions)
        next_states = np.asarray(self._next_states)
        rewards = np.asarray(self._rewards)
        dones = np.asarray(self._dones)

        start_idx = 0
        for end_idx in end_indices:
            indices = np.arange(start_idx, end_idx)
            yield states[indices], actions[indices], next_states[indices], rewards[indices], dones[indices]
            start_idx = end_idx

    def random_iterator(self, batch_size):
        """
        Iterate once through all (s, a, r, s') in batches in a random order
        """
        all_indices = np.nonzero(np.logical_not(self._dones))[0]
        np.random.shuffle(all_indices)

        states = np.asarray(self._states)
        actions = np.asarray(self._actions)
        next_states = np.asarray(self._next_states)
        rewards = np.asarray(self._rewards)
        dones = np.asarray(self._dones)

        i = 0
        while i < len(all_indices):
            indices = all_indices[i:i+batch_size]

            yield states[indices], actions[indices], next_states[indices], rewards[indices], dones[indices]

            i += batch_size

    ###############
    ### Logging ###
    ###############

    def log(self):
        end_idxs = np.nonzero(self._dones)[0] + 1

        returns = []

        start_idx = 0
        for end_idx in end_idxs:
            rewards = self._rewards[start_idx:end_idx]
            returns.append(np.sum(rewards))

            start_idx = end_idx

        logger.record_tabular('ReturnAvg', np.mean(returns))
        logger.record_tabular('ReturnStd', np.std(returns))
        logger.record_tabular('ReturnMin', np.min(returns))
        logger.record_tabular('ReturnMax', np.max(returns))

##################
### Tensorflow ###
##################

def build_mlp(input_layer,
              output_dim,
              scope,
              n_layers=1,
              hidden_dim=500,
              activation=tf.nn.relu,
              output_activation=None,
              reuse=False):
    layer = input_layer
    with tf.variable_scope(scope, reuse=reuse):
        for _ in range(n_layers):
            layer = tf.layers.dense(layer, hidden_dim, activation=activation)
        layer = tf.layers.dense(layer, output_dim, activation=output_activation)
    return layer

def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def unnormalize(x, mean, std):
    return x * std + mean

################
### Policies ###
################

class RandomPolicy(object):

    def __init__(self, env):
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high

    def get_action(self, state):
        return np.random.uniform(self._action_space_low, self._action_space_high)

