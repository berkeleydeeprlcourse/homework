import os

import numpy as np
import matplotlib.pyplot as plt

from model_based_policy import ModelBasedPolicy
import utils
from logger import logger
from timer import timeit


class ModelBasedRL(object):

    def __init__(self,
                 env,
                 num_init_random_rollouts=10,
                 max_rollout_length=500,
                 num_onplicy_iters=10,
                 num_onpolicy_rollouts=10,
                 training_epochs=60,
                 training_batch_size=512,
                 render=False,
                 mpc_horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1):
        self._env = env
        self._max_rollout_length = max_rollout_length
        self._num_onpolicy_iters = num_onplicy_iters
        self._num_onpolicy_rollouts = num_onpolicy_rollouts
        self._training_epochs = training_epochs
        self._training_batch_size = training_batch_size
        self._render = render

        logger.info('Gathering random dataset')
        self._random_dataset = self._gather_rollouts(utils.RandomPolicy(env),
                                                     num_init_random_rollouts)

        logger.info('Creating policy')
        self._policy = ModelBasedPolicy(env,
                                        self._random_dataset,
                                        horizon=mpc_horizon,
                                        num_random_action_selection=num_random_action_selection)

        timeit.reset()
        timeit.start('total')

    def _gather_rollouts(self, policy, num_rollouts):
        dataset = utils.Dataset()

        for _ in range(num_rollouts):
            state = self._env.reset()
            done = False
            t = 0
            while not done:
                if self._render:
                    timeit.start('render')
                    self._env.render()
                    timeit.stop('render')
                timeit.start('get action')
                action = policy.get_action(state)
                timeit.stop('get action')
                timeit.start('env step')
                next_state, reward, done, _ = self._env.step(action)
                timeit.stop('env step')
                done = done or (t >= self._max_rollout_length)
                dataset.add(state, action, next_state, reward, done)

                state = next_state
                t += 1

        return dataset

    def _train_policy(self, dataset):
        """
        Train the model-based policy

        implementation details:
            (a) Train for self._training_epochs number of epochs
            (b) The dataset.random_iterator(...)  method will iterate through the dataset once in a random order
            (c) Use self._training_batch_size for iterating through the dataset
            (d) Keep track of the loss values by appending them to the losses array
        """
        timeit.start('train policy')

        losses = []
        ### PROBLEM 1
        ### YOUR CODE HERE
        raise NotImplementedError

        logger.record_tabular('TrainingLossStart', losses[0])
        logger.record_tabular('TrainingLossFinal', losses[-1])

        timeit.stop('train policy')

    def _log(self, dataset):
        timeit.stop('total')
        dataset.log()
        logger.dump_tabular(print_func=logger.info)
        logger.debug('')
        for line in str(timeit).split('\n'):
            logger.debug(line)
        timeit.reset()
        timeit.start('total')

    def run_q1(self):
        """
        Train on a dataset, and see how good the learned dynamics model's predictions are.

        implementation details:
            (i) Train using the self._random_dataset
            (ii) For each rollout, use the initial state and all actions to predict the future states.
                 Store these predicted states in the pred_states list.
                 NOTE: you should *not* be using any of the states in states[1:]. Only use states[0]
            (iii) After predicting the future states, we have provided plotting code that plots the actual vs
                  predicted states and saves these to the experiment's folder. You do not need to modify this code.
        """
        logger.info('Training policy....')
        ### PROBLEM 1
        ### YOUR CODE HERE
        raise NotImplementedError

        logger.info('Evaluating predictions...')
        for r_num, (states, actions, _, _, _) in enumerate(self._random_dataset.rollout_iterator()):
            pred_states = []

            ### PROBLEM 1
            ### YOUR CODE HERE
            raise NotImplementedError

            states = np.asarray(states)
            pred_states = np.asarray(pred_states)

            state_dim = states.shape[1]
            rows = int(np.sqrt(state_dim))
            cols = state_dim // rows
            f, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            f.suptitle('Model predictions (red) versus ground truth (black) for open-loop predictions')
            for i, (ax, state_i, pred_state_i) in enumerate(zip(axes.ravel(), states.T, pred_states.T)):
                ax.set_title('state {0}'.format(i))
                ax.plot(state_i, color='k')
                ax.plot(pred_state_i, color='r')
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            f.savefig(os.path.join(logger.dir, 'prediction_{0:03d}.jpg'.format(r_num)), bbox_inches='tight')

        logger.info('All plots saved to folder')

    def run_q2(self):
        """
        Train the model-based policy on a random dataset, and evaluate the performance of the resulting policy
        """
        logger.info('Random policy')
        self._log(self._random_dataset)

        logger.info('Training policy....')
        ### PROBLEM 2
        ### YOUR CODE HERE
        raise NotImplementedError

        logger.info('Evaluating policy...')
        ### PROBLEM 2
        ### YOUR CODE HERE
        raise NotImplementedError

        logger.info('Trained policy')
        self._log(eval_dataset)

    def run_q3(self):
        """
        Starting with the random dataset, train the policy on the dataset, gather rollouts with the policy,
        append the new rollouts to the existing dataset, and repeat
        """
        dataset = self._random_dataset

        itr = -1
        logger.info('Iteration {0}'.format(itr))
        logger.record_tabular('Itr', itr)
        self._log(dataset)

        for itr in range(self._num_onpolicy_iters + 1):
            logger.info('Iteration {0}'.format(itr))
            logger.record_tabular('Itr', itr)

            ### PROBLEM 3
            ### YOUR CODE HERE
            logger.info('Training policy...')
            raise NotImplementedError

            ### PROBLEM 3
            ### YOUR CODE HERE
            logger.info('Gathering rollouts...')
            raise NotImplementedError

            ### PROBLEM 3
            ### YOUR CODE HERE
            logger.info('Appending dataset...')
            raise NotImplementedError

            self._log(new_dataset)
