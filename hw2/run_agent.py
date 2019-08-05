"""
## Running trained agent
After running `train_pg_f18.py` with a specific setting (gym environment, metaprameters) a new directory will
be added under `data` with the following structure:

    args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

Under this directory, there are multiple (exact number is set by 'n_experiments' param) trained agents.
In order to visualize (render) these agents behavior, run the `run_agent.py` script and specify the number of iterations (-n option). For example:

> python run_agent.py "data/hc_b4000_r0.01_RoboschoolInvertedPendulum-v1_21-07-2019_08-42-10/1" -n 3

"""
import os
import json
import pickle
import gym
import numpy as np
import tensorflow as tf
from train_pg_f18 import Agent


PARAMS_FILE = "params.json"
VARS_FILE = "vars.pkl"


def load_params(filename):
    """
    Load the 'params.json' file.

    A simple json.loads() call does not work here because the file was saved with a special separators.

    :param filename: str
    :return: dict
    """
    with open(filename, 'r') as file:
        data = file.read().replace(',\n', ',').replace('\t:\t', ':').replace('\n', '')

    return json.loads(data)


def load_pickle(filename, mode='rb'):
    with open(filename, mode=mode) as f:
        return pickle.load(f)


def load_agent_and_env(model_dir):
    """
    Load an agent with its pre-trained model and the relevant environment

    Most of the code here is taken from train_pg_f18.py::train_PG() function

    :param model_dir: str (full directory path to the 'params.json' and 'vars.pkl' files)
    :return: tuple (a tuple of length 2, the Agent instance and the gym env object)
    """
    # Load the params json
    params_file = os.path.join(model_dir, PARAMS_FILE)
    params = load_params(filename=params_file)
    print(params)

    # Load the model variables
    vars_filename = os.path.join(model_dir, VARS_FILE)
    model_vars = load_pickle(filename=vars_filename)
    # print(model_vars)

    # Make the gym environment
    env = gym.make(params['env_name'])

    # Set random seeds
    seed = params['seed']
    tf.set_random_seed(seed)
    np.random.seed(seed)
    #env.seed(seed)

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # ========================================================================================#
    # Initialize Agent
    # ========================================================================================#
    computation_graph_args = {
        'n_layers': params['n_layers'],
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': params['size'],
        'learning_rate': params['learning_rate'],
    }

    sample_trajectory_args = {
        'animate': params['animate'],
        'max_path_length': params['max_path_length'],
        'min_timesteps_per_batch': params['min_timesteps_per_batch'],
    }

    estimate_return_args = {
        'gamma': params['gamma'],
        'reward_to_go': params['reward_to_go'],
        'nn_baseline': params['nn_baseline'],
        'normalize_advantages': params['normalize_advantages'],
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    # Override the graph variables with the pre-trained values
    for g_var in tf.global_variables(scope=None):
        # Get the saved value and assign it to the tensor
        value = model_vars[g_var.name]
        set_variable_op = g_var.assign(value)
        agent.sess.run(set_variable_op)

    # # Validate that the assignment was successful
    # for g_var in tf.global_variables(scope=None):
    #     assert np.array_equal(g_var.eval(), model_vars[g_var.name])

    return agent, env


if __name__ == "__main__":
    """
    Example usage (after running train_pg_18.py and creating agent 'data' dirs):
    - python run_agent.py "data/hc_b4000_r0.01_RoboschoolInvertedPendulum-v1_21-07-2019_08-42-10/1" -n 3
    - python run_agent.py "data/ll_b40000_r0.005_LunarLanderContinuous-v2_21-07-2019_09-59-05/1" -n 3
    - python run_agent.py "data/hc_b50000_r0.005_RoboschoolHalfCheetah-v1_22-07-2019_20-04-48/1" -n 3
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help='A relative path to the data dir of a specific experiment. For eample: "data/ll_b40000_r0.005_LunarLanderContinuous-v2_21-07-2019_09-59-05/1"')
    parser.add_argument('--n_iter', '-n', type=int, default=3)
    args = parser.parse_args()

    # Load an agent with its pre-trained model and the relevant environment
    model_dir = args.model_dir
    agent, env = load_agent_and_env(model_dir)

    # Run an episode with this loaded agent
    for i in range(args.n_iter):
        agent.sample_trajectory(env, animate_this_episode=True)
    print("done")
