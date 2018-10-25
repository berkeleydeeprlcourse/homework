import copy
import numpy as np
import tensorflow as tf

from density_model import Density_Model
from replay import Replay_Buffer

class Exploration(object):
    def __init__(self, density_model, bonus_coeff):
        super(Exploration, self).__init__()
        self.density_model = density_model
        self.bonus_coeff = bonus_coeff

    def receive_tf_sess(self, sess):
        self.density_model.receive_tf_sess(sess)
        self.sess = sess

    def bonus_function(self, x):
        # You do not need to do anything here
        raise NotImplementedError

    def fit_density_model(self, states):
        # You do not need to do anything here
        raise NotImplementedError

    def compute_reward_bonus(self, states):
        # You do not need to do anything here
        raise NotImplementedError

    def modify_reward(self, rewards, states):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                states: (bsize, ob_dim)

            TODO:
                Use self.compute_reward_bonus to compute the reward
                bonus and then modify the rewards with the bonus
                and store that in new_rewards, which you will return
        """
        raise NotImplementedError
        bonus = None
        new_rewards = None
        return new_rewards

class DiscreteExploration(Exploration):
    def __init__(self, density_model, bonus_coeff):
        super(DiscreteExploration, self).__init__(density_model, bonus_coeff)

    def fit_density_model(self, states):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                states: (bsize, ob_dim)
        """
        raise NotImplementedError

    def bonus_function(self, count):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                count: np array (bsize)
        """
        raise NotImplementedError

    def compute_reward_bonus(self, states):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                states: (bsize, ob_dim)
        """
        count = raise NotImplementedError
        bonus = raise NotImplementedError
        return bonus


class ContinuousExploration(Exploration):
    def __init__(self, density_model, bonus_coeff, replay_size):
        super(ContinuousExploration, self).__init__(density_model, bonus_coeff)
        self.replay_buffer = Replay_Buffer(max_size=replay_size)

    def fit_density_model(self, states):
        # You do not need to do anything here
        raise NotImplementedError

    def bonus_function(self, prob):
        """
            ### PROBLEM 2
            ### YOUR CODE HERE

            args:
                prob: np array (bsize,)
        """
        raise NotImplementedError

    def compute_reward_bonus(self, states):
        """
            ### PROBLEM 2
            ### YOUR CODE HERE
        
            args:
                states: (bsize, ob_dim)
        """
        raise NotImplementedError
        prob = None
        bonus = None
        return bonus


class RBFExploration(ContinuousExploration):
    def __init__(self, density_model, bonus_coeff, replay_size):
        super(RBFExploration, self).__init__(density_model, bonus_coeff, replay_size)

    def fit_density_model(self, states):
        """
            args:
                states: (bsize, ob_dim)
        """
        self.replay_buffer.prepend(states)
        self.density_model.fit_data(self.replay_buffer.get_memory())


class ExemplarExploration(ContinuousExploration):
    def __init__(self, density_model, bonus_coeff, train_iters, bsize, replay_size):
        super(ExemplarExploration, self).__init__(density_model, bonus_coeff, replay_size)
        self.train_iters = train_iters
        self.bsize = bsize   

    def sample_idxs(self, states, batch_size):
        states = copy.deepcopy(states)
        data_size = len(states)
        pos_idxs = np.random.randint(data_size, size=batch_size)
        continue_sampling = True
        while continue_sampling:
            neg_idxs = np.random.randint(data_size, size=batch_size)
            if np.all(pos_idxs != neg_idxs):
                continue_sampling = False
        positives = np.concatenate([states[pos_idxs], states[pos_idxs]], axis=0)
        negatives = np.concatenate([states[pos_idxs], states[neg_idxs]], axis=0)
        return positives, negatives

    def sample_idxs_replay(self, states, batch_size):
        states = copy.deepcopy(states)
        data_size = len(states)
        pos_idxs = np.random.randint(data_size, size=batch_size)
        neg_idxs = np.random.randint(data_size, len(self.replay_buffer), size=batch_size)
        positives = np.concatenate([states[pos_idxs], states[pos_idxs]], axis=0)
        negatives = np.concatenate([states[pos_idxs], self.replay_buffer[neg_idxs]], axis=0)
        return positives, negatives

    def fit_density_model(self, states):
        """
            args:
                states: (bsize, ob_dim)
        """
        self.replay_buffer.prepend(states)
        for i in range(self.train_iters):
            if len(self.replay_buffer) >= 2*len(states):
                positives, negatives = self.sample_idxs_replay(states, self.bsize)
            else:
                positives, negatives = self.sample_idxs(states, self.bsize)
            labels = np.concatenate([np.ones((self.bsize, 1)), np.zeros((self.bsize, 1))], axis=0)
            ll, kl, elbo = self.density_model.update(positives, negatives, labels)
            if i % (self.train_iters/10) == 0:
                print('log likelihood\t{}\tkl divergence\t{}\t-elbo\t{}'.format(np.mean(ll), np.mean(kl), -elbo))
        return ll, kl, elbo
