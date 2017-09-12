"""
Code to load a built model and generate roll-out data
Example usage:
    python run_policy.py /tmp/model Hopper-v1 --num_rollouts 20

"""

import argparse

import gym
import numpy as np
import tensorflow as tf

from behavioral_cloning import feedforward_nn, DEFAULT_HIDDEN_LAYER_SIZE

def main(model_filepath, envname, num_rollouts, hidden_layer_size=DEFAULT_HIDDEN_LAYER_SIZE, max_timesteps=None, render=False):
    env = gym.make(envname)

    # TODO nthomas - note that this won't work for input tensors, only input vectors.
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    x = tf.placeholder(tf.float32, [None, observation_size])
    y = feedforward_nn(x, observation_size, action_size, hidden_layer_size)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_filepath)

        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # need to turn this into a vector of inputs
                action = sess.run(y, {x: obs[None,:]})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        # policy observed states
        # policy_observed_data = {'observations': np.array(observations),
        #                         'actions': np.array(actions)}
        # save_obj(expert_data, expert_data_filename)
    return returns


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('model_filepath', type=str)
  parser.add_argument('envname', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--max_timesteps', type=int)
  parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
  parser.add_argument('--hidden_layer_size', type=int, default=DEFAULT_HIDDEN_LAYER_SIZE)
  args = parser.parse_args()

  main(args.model_filepath,
       args.envname,
       args.num_rollouts,
       args.hidden_layer_size,
       args.max_timesteps,
       args.render)
