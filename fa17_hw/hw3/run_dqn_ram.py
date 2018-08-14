import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
from atari_wrappers import *


def atari_model(ram_in, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = ram_in
        #out = tf.concat(1,(ram_in[:,4:5],ram_in[:,8:9],ram_in[:,11:13],ram_in[:,21:22],ram_in[:,50:51], ram_in[:,60:61],ram_in[:,64:65]))
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0 
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 0.2),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=1,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(seed):
    env = gym.make('Pong-ram-v0')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind_ram(env)

    return env

def main():
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(seed)
    session = get_session()
    atari_learn(env, session, num_timesteps=int(4e7))

if __name__ == "__main__":
    main()
