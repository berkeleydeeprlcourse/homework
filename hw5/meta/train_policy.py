"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
Adapted for use in CS294-112 Fall 2018 HW5 by Kate Rakelly and Michael Chang
"""
import numpy as np
import pdb
import random
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process

from replay_buffer import ReplayBuffer, PPOReplayBuffer

from point_mass import PointEnv
from point_mass_observed import ObservedPointEnv

#============================================================================================#
# Utilities
#============================================================================================#
def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """
    minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)

def build_mlp(x, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None, regularizer=None):
    """
    builds a feedforward neural network

    arguments:
        x: placeholder variable for the state (batch_size, input_size)
        regularizer: regularization for weights
        (see `build_policy()` for rest)

    returns:
        output placeholder of the network (the result of a forward pass)
    """
    i = 0
    for i in range(n_layers):
        x = tf.layers.dense(inputs=x,units=size, activation=activation, name='fc{}'.format(i), kernel_regularizer=regularizer, bias_regularizer=regularizer)

    x = tf.layers.dense(inputs=x, units=output_size, activation=output_activation, name='fc{}'.format(i + 1), kernel_regularizer=regularizer, bias_regularizer=regularizer)
    return x

def build_rnn(x, h, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None, regularizer=None):
    """
    builds a gated recurrent neural network
    inputs are first embedded by an MLP then passed to a GRU cell

    make MLP layers with `size` number of units
    make the GRU with `output_size` number of units
    use `activation` as the activation function for both MLP and GRU

    arguments:
        (see `build_policy()`)

    hint: use `build_mlp()`
    """
    #====================================================================================#
    #                           ----------PROBLEM 2----------
    #====================================================================================#
    # YOUR CODE HERE

def build_policy(x, h, output_size, scope, n_layers, size, gru_size, recurrent=True, activation=tf.tanh, output_activation=None):
    """
    build recurrent policy

    arguments:
        x: placeholder variable for the input, which has dimension (batch_size, history, input_size)
        h: placeholder variable for the hidden state, which has dimension (batch_size, gru_size)
        output_size: size of the output layer, same as action dimension
        scope: variable scope of the network
        n_layers: number of hidden layers (not counting recurrent units)
        size: dimension of the hidden layer in the encoder
        gru_size: dimension of the recurrent hidden state if there is one
        recurrent: if the network should be recurrent or feedforward
        activation: activation of the hidden layers
        output_activation: activation of the ouput layers

    returns:
        output placeholder of the network (the result of a forward pass)

    n.b. we predict both the mean and std of the gaussian policy, and we don't want the std to start off too large
    initialize the last layer of the policy with a guassian init of mean 0 and std 0.01
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if recurrent:
            x, h = build_rnn(x, h, gru_size, scope, n_layers, size, activation=activation, output_activation=activation)
        else:
            x = tf.reshape(x, (-1, x.get_shape()[1]*x.get_shape()[2]))
            x = build_mlp(x, gru_size, scope, n_layers + 1, size, activation=activation, output_activation=activation)
        x = tf.layers.dense(x, output_size, activation=output_activation, kernel_initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01), bias_initializer=tf.zeros_initializer(), name='decoder')
    return x, h

def build_critic(x, h, output_size, scope, n_layers, size, gru_size, recurrent=True, activation=tf.tanh, output_activation=None, regularizer=None):
    """
    build recurrent critic

    arguments:
        regularizer: regularization for weights
        (see `build_policy()` for rest)

    n.b. the policy and critic should not share weights
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if recurrent:
            x, h = build_rnn(x, h, gru_size, scope, n_layers, size, activation=activation, output_activation=output_activation, regularizer=regularizer)
        else:
            x = tf.reshape(x, (-1, x.get_shape()[1]*x.get_shape()[2]))
            x = build_mlp(x, gru_size, scope, n_layers + 1, size, activation=activation, output_activation=activation, regularizer=regularizer)
        x = tf.layers.dense(x, output_size, activation=output_activation, name='decoder', kernel_regularizer=regularizer, bias_regularizer=regularizer)
    return x

def pathlength(path):
    return len(path["reward"])

def discounted_return(reward, gamma):
    discounts = gamma**np.arange(len(reward))
    return sum(discounts * reward)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)


class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.task_dim = computation_graph_args['task_dim']
        self.reward_dim = 1
        self.terminal_dim = 1

        self.meta_ob_dim = self.ob_dim + self.ac_dim + self.reward_dim + self.terminal_dim
        self.scope  = 'continuous_logits'
        self.size = computation_graph_args['size']
        self.gru_size = computation_graph_args['gru_size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.history = computation_graph_args['history']
        self.num_value_iters = computation_graph_args['num_value_iters']
        self.l2reg = computation_graph_args['l2reg']
        self.recurrent = computation_graph_args['recurrent']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.nn_critic = estimate_return_args['nn_critic']
        self.normalize_advantages = estimate_return_args['normalize_advantages']

        self.replay_buffer = ReplayBuffer(100000, [self.history, self.meta_ob_dim], [self.ac_dim], self.gru_size, self.task_dim)
        self.val_replay_buffer = ReplayBuffer(100000, [self.history, self.meta_ob_dim], [self.ac_dim], self.gru_size, self.task_dim)

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True # may need if using GPU
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    def define_placeholders(self):
        """
        placeholders for batch batch observations / actions / advantages in policy gradient
        loss function.
        see Agent.build_computation_graph for notation

        returns:
            sy_ob_no: placeholder for meta-observations
            sy_ac_na: placeholder for actions
            sy_adv_n: placeholder for advantages
            sy_hidden: placeholder for RNN hidden state

            (PPO stuff)
            sy_lp_n: placeholder for pre-computed log-probs
            sy_fixed_lp_n: placeholder for pre-computed old log-probs
        """
        sy_ob_no = tf.placeholder(shape=[None, self.history, self.meta_ob_dim], name="ob", dtype=tf.float32)
        sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)
        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        sy_hidden = tf.placeholder(shape=[None, self.gru_size], name="hidden", dtype=tf.float32)

        sy_lp_n = tf.placeholder(shape=[None], name="logprob", dtype=tf.float32)
        sy_fixed_lp_n = tf.placeholder(shape=[None], name="fixed_logprob", dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n, sy_hidden, sy_lp_n, sy_fixed_lp_n

    def policy_forward_pass(self, sy_ob_no, sy_hidden):
        """
        constructs the symbolic operation for the policy network outputs,
        which are the parameters of the policy distribution p(a|s)

        arguments:
            sy_ob_no: (batch_size, self.history, self.meta_ob_dim)
            sy_hidden: (batch_size, self.gru_size)

        returns:
            the parameters of the policy.

            the parameters are a tuple (mean, log_std) of a Gaussian
                distribution over actions. log_std should just be a trainable
                variable, not a network output.
                sy_mean: (batch_size, self.ac_dim)
                sy_logstd: (batch_size, self.ac_dim)

        """
        # ac_dim * 2 because we predict both mean and std
        sy_policy_params, sy_hidden = build_policy(sy_ob_no, sy_hidden, self.ac_dim*2, self.scope, n_layers=self.n_layers, size=self.size, gru_size=self.gru_size, recurrent=self.recurrent)
        return (sy_policy_params, sy_hidden)

    def sample_action(self, policy_parameters):
        """
        constructs a symbolic operation for stochastically sampling from the policy
        distribution

        arguments:
            policy_parameters
                mean, log_std) of a Gaussian distribution over actions
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (batch_size, self.ac_dim)

        returns:
            sy_sampled_ac:
                (batch_size, self.ac_dim)
        """
        sy_mean, sy_logstd = policy_parameters
        sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * tf.random_normal(tf.shape(sy_mean), 0, 1)
        return sy_sampled_ac

    def get_log_prob(self, policy_parameters, sy_ac_na):
        """
        constructs a symbolic operation for computing the log probability of a set of actions
        that were actually taken according to the policy

        arguments:
            policy_parameters
                mean, log_std) of a Gaussian distribution over actions
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (batch_size, self.ac_dim)

            sy_ac_na: (batch_size, self.ac_dim)

        returns:
            sy_lp_n: (batch_size)

        """
        sy_mean, sy_logstd = policy_parameters
        sy_lp_n = tfp.distributions.MultivariateNormalDiag(
            loc=sy_mean, scale_diag=tf.exp(sy_logstd)).log_prob(sy_ac_na)
        return sy_lp_n

    def build_computation_graph(self):
        """
        notes on notation:

        Symbolic variables have the prefix sy_, to distinguish them from the numerical values
        that are computed later in the function

        prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
        is None

        ----------------------------------------------------------------------------------
        loss: a function of self.sy_lp_n and self.sy_adv_n that we will differentiate
            to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n, self.sy_hidden, self.sy_lp_n, self.sy_fixed_lp_n = self.define_placeholders()

        # The policy takes in an observation and produces a distribution over the action space
        policy_outputs = self.policy_forward_pass(self.sy_ob_no, self.sy_hidden)
        self.policy_parameters = policy_outputs[:-1]

        # unpack mean and variance
        self.policy_parameters = tf.split(self.policy_parameters[0], 2, axis=1)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_lp_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        # PPO critic update
        critic_regularizer = tf.contrib.layers.l2_regularizer(1e-3) if self.l2reg else None
        self.critic_prediction = tf.squeeze(build_critic(self.sy_ob_no, self.sy_hidden, 1, 'critic_network', n_layers=self.n_layers, size=self.size, gru_size=self.gru_size, recurrent=self.recurrent, regularizer=critic_regularizer))
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)
        self.critic_loss = tf.losses.mean_squared_error(self.sy_target_n, self.critic_prediction)
        self.critic_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_network')
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

        # PPO actor update
        self.sy_fixed_log_prob_n = tf.placeholder(shape=[None], name="fixed_log_prob", dtype=tf.float32)
        self.policy_surr_loss = self.ppo_loss(self.sy_lp_n, self.sy_fixed_lp_n, self.sy_adv_n)
        self.policy_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.policy_update_op = minimize_and_clip(optimizer, self.policy_surr_loss, var_list=self.policy_weights, clip_val=40)

    def sample_trajectories(self, itr, env, min_timesteps, is_evaluation=False):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        stats = []
        while True:
            animate_this_episode=(len(stats)==0 and (itr % 10 == 0) and self.animate)
            steps, s = self.sample_trajectory(env, animate_this_episode, is_evaluation=is_evaluation)
            stats += s
            timesteps_this_batch += steps
            if timesteps_this_batch > min_timesteps:
                break
        return stats, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode, is_evaluation):
        """
        sample a task, then sample trajectories from that task until either
        max(self.history, self.max_path_length) timesteps have been sampled

        construct meta-observations by concatenating (s, a, r, d) into one vector
        inputs to the policy should have the shape (batch_size, self.history, self.meta_ob_dim)
        zero pad the input to maintain a consistent input shape

        add the entire input as observation to the replay buffer, along with a, r, d
        samples will be drawn from the replay buffer to update the policy

        arguments:
            env: the env to sample trajectories from
            animate_this_episode: if True then render
            val: whether this is training or evaluation
        """
        env.reset_task(is_evaluation=is_evaluation)
        stats = []
        #====================================================================================#
        #                           ----------PROBLEM 1----------
        #====================================================================================#
        ep_steps = 0
        steps = 0

        num_samples = max(self.history, self.max_path_length + 1)
        meta_obs = np.zeros((num_samples + self.history + 1, self.meta_ob_dim))
        rewards = []

        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)

            if ep_steps == 0:
                ob = env.reset()
                # first meta ob has only the observation
                # set a, r, d to zero, construct first meta observation in meta_obs
                # YOUR CODE HERE

                steps += 1

            # index into the meta_obs array to get the window that ends with the current timestep
            # please name the windowed observation `in_` for compatibilty with the code that adds to the replay buffer (lines 418, 420)
            # YOUR CODE HERE

            hidden = np.zeros((1, self.gru_size), dtype=np.float32)

            # get action from the policy
            # YOUR CODE HERE

            # step the environment
            # YOUR CODE HERE

            ep_steps += 1

            done = bool(done) or ep_steps == self.max_path_length
            # construct the meta-observation and add it to meta_obs
            # YOUR CODE HERE

            rewards.append(rew)
            steps += 1

            # add sample to replay buffer
            if is_evaluation:
                self.val_replay_buffer.add_sample(in_, ac, rew, done, hidden, env._goal)
            else:
                self.replay_buffer.add_sample(in_, ac, rew, done, hidden, env._goal)

            # start new episode
            if done:
                # compute stats over trajectory
                s = dict()
                s['rewards']= rewards[-ep_steps:]
                s['ep_len'] = ep_steps
                stats.append(s)
                ep_steps = 0

            if steps >= num_samples:
                break

        return steps, stats

    def compute_advantage(self, ob_no, re_n, hidden, masks, tau=0.95):
        """
        computes generalized advantage estimation (GAE).

        arguments:
            ob_no: (bsize, history, ob_dim)
            rewards: (bsize,)
            masks: (bsize,)
            values: (bsize,)
            gamma: scalar
            tau: scalar

        output:
            advantages: (bsize,)
            returns: (bsize,)

        requires:
            self.gamma
        """
        bsize = len(re_n)
        rewards = np.squeeze(re_n)
        masks = np.squeeze(masks)
        values = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: ob_no, self.sy_hidden: hidden})[:,None]
        gamma = self.gamma

        assert rewards.shape == masks.shape == (bsize,)
        assert values.shape == (bsize, 1)

        bsize = len(rewards)
        returns = np.empty((bsize,))
        deltas = np.empty((bsize,))
        advantages = np.empty((bsize,))

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(bsize)):
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]

        advantages = (advantages - np.mean(advantages, axis=0)) / np.std(advantages, axis=0)
        return advantages, returns


    def estimate_return(self, ob_no, re_n, hidden, masks):
        """
        estimates the returns over a set of trajectories.

        let sum_of_path_lengths be the sum of the lengths of the paths sampled from
            Agent.sample_trajectories
        let num_paths be the number of paths sampled from Agent.sample_trajectories

        arguments:
            ob_no: shape: (sum_of_path_lengths, history, meta_obs_dim)
            re_n: length: num_paths. Each element in re_n is a numpy array
                containing the rewards for the particular path
            hidden: hidden state of recurrent policy
            masks: terminals masks

        returns:
            q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values
                whose length is the sum of the lengths of the paths
            adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                advantages whose length is the sum of the lengths of the paths
        """
        adv_n, q_n = self.compute_advantage(ob_no, re_n, hidden, masks)
        return q_n, adv_n

    def update_parameters(self, ob_no, hidden, ac_na, fixed_log_probs, q_n, adv_n):
        """
        update the parameters of the policy and the critic,
        with PPO update

        arguments:
            ob_no: (minibsize, history, meta_obs_dim)
            hidden: shape: (minibsize, self.gru_size)
            ac_na: (minibsize)
            fixed_log_probs: (minibsize)
            adv_n: shape: (minibsize)
            q_n: shape: (sum_of_path_lengths)

        returns:
            nothing

        """
        self.update_critic(ob_no, hidden, q_n)
        self.update_policy(ob_no, hidden, ac_na, fixed_log_probs, adv_n)

    def update_critic(self, ob_no, hidden, q_n):
        """
        given:
            self.num_value_iters
            self.l2_reg

        arguments:
            ob_no: (minibsize, history, meta_obs_dim)
            hidden: (minibsize, self.gru_size)
            q_n: (minibsize)

        requires:
            self.num_value_iters
        """
        target_n = (q_n - np.mean(q_n))/(np.std(q_n)+1e-8)
        for k in range(self.num_value_iters):
            critic_loss, _ = self.sess.run(
                [self.critic_loss, self.critic_update_op],
                feed_dict={self.sy_target_n: target_n, self.sy_ob_no: ob_no, self.sy_hidden: hidden})
        return critic_loss

    def update_policy(self, ob_no, hidden, ac_na, fixed_log_probs, advantages):
        '''
        arguments:
            fixed_log_probs: (minibsize)
            advantages: (minibsize)
            hidden: (minibsize, self.gru_size)
        '''
        policy_surr_loss, _ = self.sess.run(
            [self.policy_surr_loss, self.policy_update_op],
            feed_dict={self.sy_ob_no: ob_no, self.sy_hidden: hidden, self.sy_ac_na: ac_na, self.sy_fixed_lp_n: fixed_log_probs, self.sy_adv_n: advantages})
        return policy_surr_loss

    def ppo_loss(self, log_probs, fixed_log_probs, advantages, clip_epsilon=0.1, entropy_coeff=1e-4):
        """
        given:
            clip_epsilon

        arguments:
            advantages (mini_bsize,)
            states (mini_bsize,)
            actions (mini_bsize,)
            fixed_log_probs (mini_bsize,)

        intermediate results:
            states, actions --> log_probs
            log_probs, fixed_log_probs --> ratio
            advantages, ratio --> surr1
            ratio, clip_epsilon, advantages --> surr2
            surr1, surr2 --> policy_surr_loss
        """
        ratio = tf.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, clip_value_min=1.0-clip_epsilon, clip_value_max=1.0+clip_epsilon) * advantages
        policy_surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        probs = tf.exp(log_probs)
        entropy = tf.reduce_sum(-(log_probs * probs))
        policy_surr_loss -= entropy_coeff * entropy
        return policy_surr_loss


def train_PG(
        exp_name,
        env_name,
        n_iter,
        gamma,
        min_timesteps_per_batch,
        mini_batch_size,
        max_path_length,
        learning_rate,
        num_ppo_updates,
        num_value_iters,
        animate,
        logdir,
        normalize_advantages,
        nn_critic,
        seed,
        n_layers,
        size,
        gru_size,
        history,
        num_tasks,
        l2reg,
        recurrent,
        ):

    start = time.time()

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    envs = {'pm': PointEnv,
            'pm-obs': ObservedPointEnv,
            }
    env = envs[env_name](num_tasks)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    task_dim = len(env._goal) # rude, sorry

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'task_dim': task_dim,
        'size': size,
        'gru_size': gru_size,
        'learning_rate': learning_rate,
        'history': history,
        'num_value_iters': num_value_iters,
        'l2reg': l2reg,
        'recurrent': recurrent,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'nn_critic': nn_critic,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)

    # build computation graph
    agent.build_computation_graph()


    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    #========================================================================================#
    # Training Loop
    #========================================================================================#
    def unpack_sample(data):
        '''
        unpack a sample from the replay buffer
        '''
        ob = data["observations"]
        ac = data["actions"]
        re = data["rewards"]
        hi = data["hiddens"]
        ma = 1 - data["terminals"]
        return ob, ac, re, hi, ma

    # construct PPO replay buffer, perhaps rude to do outside the agent
    ppo_buffer = PPOReplayBuffer(agent.replay_buffer)

    total_timesteps = 0
    for itr in range(n_iter):
        # for PPO: flush the replay buffer!
        ppo_buffer.flush()

        # sample trajectories to fill agent's replay buffer
        print("********** Iteration %i ************"%itr)
        stats = []
        for _ in range(num_tasks):
            s, timesteps_this_batch = agent.sample_trajectories(itr, env, min_timesteps_per_batch)
            total_timesteps += timesteps_this_batch
            stats += s

        # compute the log probs, advantages, and returns for all data in agent's buffer
        # store in ppo buffer for use in multiple ppo updates
        # TODO: should move inside the agent probably
        data = agent.replay_buffer.all_batch()
        ob_no, ac_na, re_n, hidden, masks = unpack_sample(data)
        fixed_log_probs = agent.sess.run(agent.sy_lp_n,
            feed_dict={agent.sy_ob_no: ob_no, agent.sy_hidden: hidden, agent.sy_ac_na: ac_na})
        q_n, adv_n = agent.estimate_return(ob_no, re_n, hidden, masks)

        ppo_buffer.add_samples(fixed_log_probs, adv_n, q_n)

        # update with mini-batches sampled from ppo buffer
        for _ in range(num_ppo_updates):

            data = ppo_buffer.random_batch(mini_batch_size)

            ob_no, ac_na, re_n, hidden, masks = unpack_sample(data)
            fixed_log_probs = data["log_probs"]
            adv_n = data["advantages"]
            q_n = data["returns"]

            log_probs = agent.sess.run(agent.sy_lp_n,
                feed_dict={agent.sy_ob_no: ob_no, agent.sy_hidden: hidden, agent.sy_ac_na: ac_na})

            agent.update_parameters(ob_no, hidden, ac_na, fixed_log_probs, q_n, adv_n)

        # compute validation statistics
        print('Validating...')
        val_stats = []
        for _ in range(num_tasks):
            vs, timesteps_this_batch = agent.sample_trajectories(itr, env, min_timesteps_per_batch // 10, is_evaluation=True)
            val_stats += vs

        # save trajectories for viz
        with open("output/{}-epoch{}.pkl".format(exp_name, itr), 'wb') as f:
            pickle.dump(agent.val_replay_buffer.all_batch(), f, pickle.HIGHEST_PROTOCOL)
        agent.val_replay_buffer.flush()

        # Log TRAIN diagnostics
        returns = [sum(s["rewards"]) for s in stats]
        final_rewards = [s["rewards"][-1] for s in stats]
        ep_lengths = [s['ep_len'] for s in stats]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("FinalReward", np.mean(final_rewards))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)

        # Log VAL diagnostics
        val_returns = [sum(s["rewards"]) for s in val_stats]
        val_final_rewards = [s["rewards"][-1] for s in val_stats]
        logz.log_tabular("ValAverageReturn", np.mean(val_returns))
        logz.log_tabular("ValFinalReward", np.mean(val_final_rewards))

        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-pb', type=int, default=10000)
    parser.add_argument('--mini_batch_size', '-mpb', type=int, default=64)
    parser.add_argument('--num_tasks', '-nt', type=int, default=1)
    parser.add_argument('--ep_len', '-ep', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4)
    parser.add_argument('--num_value_iters', '-nvu', type=int, default=1)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_critic', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--gru_size', '-rs', type=int, default=32)
    parser.add_argument('--history', '-ho', type=int, default=1)
    parser.add_argument('--l2reg', '-reg', action='store_true')
    parser.add_argument('--recurrent', '-rec', action='store_true')
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size // args.num_tasks,
                mini_batch_size=args.mini_batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                num_ppo_updates=(args.batch_size // args.mini_batch_size) * 5,
                num_value_iters=args.num_value_iters,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_critic=args.nn_critic,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                gru_size=args.gru_size,
                history=args.history,
                num_tasks=args.num_tasks,
                l2reg=args.l2reg,
                recurrent=args.recurrent,
                )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
