import random
import numpy as np
import copy

class Replay_Buffer(object):
    def __init__(self, max_size=np.inf):
        self.memory = []
        self.max_size = int(max_size)

    def adjust_size(self):
        if len(self.memory) > self.max_size:
            diff = int(len(self.memory) - self.max_size)
            self.memory = self.memory[:-diff]  # FIFO
            print('Adjusted replay size')

    def prepend(self, x):
        # assume x is a list of states
        self.memory = list(x) + self.memory
        self.adjust_size()

    def sample(self, batch_size):
        random_batch = random.sample(self.memory, batch_size)
        return random_batch

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, indices):
        return copy.deepcopy(np.array([self.memory[i] for i in indices]))

    def get_memory(self):
        return copy.deepcopy(self.memory)

    def clear_buffer(self):
        del self.memory[:]