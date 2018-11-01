import numpy as np

class ReplayBuffer(object):
    '''
    minimalistic replay buffer

    a sample consists of
     - observation
     - action
     - reward
     - terminal
     - hidden state for recurrent policy

     it is memory inefficient to store windowed observations this way
     so do not run on tasks with large observations (e.g. from vision)
    '''

    def __init__(self, max_size, ob_dim, ac_dim, hidden_dim, task_dim):
        self.max_size = max_size
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.flush()

    def flush(self):
        '''
        set buffer to empty
        '''
        self._observations = np.zeros((self.max_size, *self.ob_dim))
        self._actions = np.zeros((self.max_size, *self.ac_dim))
        self._rewards = np.zeros((self.max_size, 1))
        self._terminals = np.zeros((self.max_size, 1))
        self._hiddens = np.zeros((self.max_size, self.hidden_dim))
        self._tasks = np.zeros((self.max_size, self.task_dim))
        self._top = 0
        self._size = 0

    def _advance(self):
        '''
        move pointer to top of buffer
        if end of buffer is reached, overwrite oldest data
        '''
        self._top = (self._top + 1) % self.max_size
        if self._size < self.max_size:
            self._size += 1

    def add_sample(self, ob, ac, re, te, hi, task):
        '''
        add sample to buffer
        '''
        self._observations[self._top] = ob
        self._actions[self._top] = ac
        self._rewards[self._top] = re
        self._terminals[self._top] = te
        self._hiddens[self._top] = hi
        self._tasks[self._top] = task

        self._advance()

    def get_samples(self, indices):
        '''
        return buffer data indexed by `indices`
        '''
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            hiddens=self._hiddens[indices],
            tasks=self._tasks[indices],
        )

    def random_batch(self, batch_size):
        '''
        return random sample of `batch_size` transitions
        '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.get_samples(indices)

    def all_batch(self):
        '''
        return all data in the buffer
        '''
        indices = list(range(self._size))
        return self.get_samples(indices)

    def num_steps_can_sample(self):
        return self._size



class PPOReplayBuffer(object):
    '''
    replay buffer for PPO algorithm
    store fixed log probs, advantages, and returns for use in multiple updates

    n.b. samples must be added as a batch, and we assume that the
    batch is the same size as that of the simple buffer
    '''

    def __init__(self, simple_buffer):
        self.simple_buffer = simple_buffer
        self.max_size = self.simple_buffer.max_size
        self.flush()

    def flush(self):
        self.simple_buffer.flush()
        self._log_probs = np.zeros((self.max_size, 1))
        self._advantages = np.zeros((self.max_size, 1))
        self._returns = np.zeros((self.max_size, 1))

    def add_samples(self, lp, adv, ret):
        self._log_probs = lp
        self._advantages = adv
        self._returns = ret

    def get_samples(self, indices):
        return dict(
            log_probs = self._log_probs[indices],
            advantages = self._advantages[indices],
            returns = self._returns[indices],
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self.simple_buffer._size, batch_size)
        simple = self.simple_buffer.get_samples(indices)
        ppo = self.get_samples(indices)
        return {**simple, **ppo}
