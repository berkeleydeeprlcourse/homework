import gym
from gym.envs.registration import EnvSpec
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm

class Env(object):
    def __init__(self):
        super(Env, self).__init__()

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def seed(self, seed):
        raise NotImplementedError

class PointMass(Env):
    def __init__(self, max_episode_steps_coeff=1, scale=20, goal_padding=2.0):
        super(PointMass, self).__init__()
        # define scale such that the each square in the grid is 1 x 1
        self.scale = int(scale)
        self.grid_size = self.scale * self.scale
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]))
        self.action_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([np.inf, np.inf]))
        self.goal_padding = goal_padding
        self.spec = EnvSpec(id='PointMass-v0', max_episode_steps=int(max_episode_steps_coeff*self.scale))

    def reset(self):
        plt.close()
        self.state = np.array([self.goal_padding, self.goal_padding])
        state = self.state/self.scale
        return state

    def step(self, action):
        x, y = action

        # next state
        new_x = self.state[0]+x
        new_y = self.state[1]+y
        if new_x < 0:
            new_x = 0
        if new_x > self.scale:
            new_x = self.scale
        if new_y < 0:
            new_y = 0
        if new_y > self.scale:
            new_y = self.scale
        self.state = np.array([new_x, new_y])
        state = self.state/self.scale

        # reward
        reg_term = -0.01*np.sum(action**2)

        threshold = self.scale - self.goal_padding
        if new_x > threshold and new_y > threshold:
            reward = 10 + reg_term
        else:
            reward = 0 + reg_term

        # done
        done = False

        return state, reward, done, None

    def preprocess(self, state):
        scaled_state = self.scale * state
        x_floor, y_floor = np.floor(scaled_state)
        assert x_floor <= self.scale
        assert y_floor <= self.scale
        if x_floor == self.scale:
            x_floor -= 1
        if y_floor == self.scale:
            y_floor -= 1
        index = self.scale*x_floor + y_floor
        return index

    def unprocess(self, index):
        x_floor = index // self.scale
        y_floor = index % self.scale
        unscaled_state = np.array([x_floor, y_floor])/self.scale
        return unscaled_state

    def seed(self, seed):
        pass

    def render(self):
        # create a grid
        states = [self.state/self.scale]
        indices = np.array([int(self.preprocess(s)) for s in states])
        a = np.zeros(self.grid_size)
        for i in indices:
            a[i] += 1
        max_freq = np.max(a)
        a/=float(max_freq)  # normalize
        a = np.reshape(a, (self.scale, self.scale))
        ax = sns.heatmap(a)
        plt.draw()
        plt.pause(0.001)
        plt.clf()        

    def visualize(self, states, itr, dirname):
        if states is None:
            states = np.load(os.path.join(dirname, '{}.npy'.format(itr)))
        indices = np.array([int(self.preprocess(s)) for s in states])
        a = np.zeros(int(self.grid_size))
        for i in indices:
            a[i] += 1
        max_freq = np.max(a)
        a/=float(max_freq)  # normalize
        a = np.reshape(a, (self.scale, self.scale))
        ax = sns.heatmap(a)
        plt.savefig(os.path.join(dirname, '{}.png'.format(itr)))
        plt.close()

    def create_gif(self, dirname, density=False):
        images = []
        if density:
            filenames = [x for x in os.listdir(dirname) if '_density.png' in x]
            sorted_fnames = sorted(filenames, key=lambda x: int(x.split('_density.png')[0]))
        else:
            filenames = [x for x in os.listdir(dirname) if ('.png' in x and 'density' not in x)]
            sorted_fnames = sorted(filenames, key=lambda x: int(x.split('.png')[0]))
        for f in sorted_fnames:
            images.append(imageio.imread(os.path.join(dirname, f)))
        imageio.mimsave(os.path.join(dirname, 'exploration.gif'), images)

    def create_visualization(self, dirname, density=False):
        for s in os.listdir(dirname):
            for i in tqdm(range(100)):
                self.visualize(None, i, os.path.join(dirname, s))
            self.create_gif(os.path.join(dirname, str(s)))

def debug():
    logdir = 'pm_debug'
    os.mkdir(logdir)
    num_episodes = 10
    num_steps_per_epoch = 20

    env = PointMass()
    for epoch in range(num_episodes):
        states = []
        state = env.reset()
        for i in range(num_steps_per_epoch):
            action = np.random.rand(2)
            state, reward, done, _ = env.step(action)
            states.append(state)
        env.visualize(np.array(states), epoch, logdir)
    env.create_gif(logdir)


if __name__ == "__main__":
    # debug()  # run this if you want to get a feel for how the PointMass environment works (make sure to comment out the code below)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', type=str)
    args = parser.parse_args()
    env = PointMass()
    env.create_visualization(args.dirname)




