"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Soroush Nasiriany, Sid Reddy, and Greg Kahn
Adapted for CS294-112 Fall 2018 with <3 by Michael Chang, some experiments by Greg Kahn, beta-tested by Sid Reddy
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process

from exploration import ExemplarExploration, DiscreteExploration, RBFExploration
from density_model import Exemplar, Histogram, RBF

#============================================================================================#
# Utilities
#============================================================================================#

def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network
        
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass) 

        Hint: use tf.layers.dense    
    """
    output_placeholder = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            output_placeholder = tf.layers.dense(output_placeholder, size, activation=activation)
        output_placeholder = tf.layers.dense(output_placeholder, output_size, activation=output_activation)
    return output_placeholder

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

#============================================================================================#
# Actor Critic
#============================================================================================#

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_advantage_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.num_target_updates = computation_graph_args['num_target_updates']
        self.num_grad_steps_per_target_update = computation_graph_args['num_grad_steps_per_target_update']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_advantage_args['gamma']
        self.normalize_advantages = estimate_advantage_args['normalize_advantages']

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        tf_config.gpu_options.allow_growth = True # may need if using GPU
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in actor critic
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 
        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n

    def policy_forward_pass(self, sy_ob_no):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_logits_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case)
                and the mean (in the continuous case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """
        if self.discrete:
            sy_logits_na = build_mlp(sy_ob_no, self.ac_dim, 'discrete_logits', n_layers=self.n_layers, size=self.size)
            return sy_logits_na
        else:
            sy_mean = build_mlp(sy_ob_no, self.ac_dim, 'continuous_logits', n_layers=self.n_layers, size=self.size)
            sy_logstd = tf.Variable(tf.zeros(self.ac_dim), name='sy_logstd')
            return (sy_mean, sy_logstd)

    def sample_action(self, policy_parameters):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

            returns:
                sy_sampled_ac: 
                    if discrete: (batch_size)
                    if continuous: (batch_size, self.ac_dim)

            Hint: for the continuous case, use the reparameterization trick:
                 The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
        
                      mu + sigma * z,         z ~ N(0, I)
        
                 This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
        """
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, num_samples=1), axis=1)
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * tf.random_normal(tf.shape(sy_mean), 0, 1)
        return sy_sampled_ac

    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

                sy_ac_na: (batch_size, self.ac_dim)

            returns:
                sy_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
                For the continuous case, use the log probability under a multivariate gaussian.
        """
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_logprob_n = tf.distributions.Categorical(logits=sy_logits_na).log_prob(sy_ac_na)
        else:
            sy_mean, sy_logstd = policy_parameters
            sy_logprob_n = tfp.distributions.MultivariateNormalDiag(
                loc=sy_mean, scale_diag=tf.exp(sy_logstd)).log_prob(sy_ac_na)  
        return sy_logprob_n

    def build_computation_graph(self):
        """
            Notes on notation:
            
            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function
            
            Prefixes and suffixes:
            ob - observation 
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)
            
            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()

        # The policy takes in an observation and produces a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        actor_loss = tf.reduce_sum(-self.sy_logprob_n * self.sy_adv_n)
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(actor_loss)

        # define the critic
        self.critic_prediction = tf.squeeze(build_mlp(
                                self.sy_ob_no,
                                1,
                                "nn_critic",
                                n_layers=self.n_layers,
                                size=self.size))
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)
        self.critic_loss = tf.losses.mean_squared_error(self.sy_target_n, self.critic_prediction)
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no : ob[None]})
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            next_obs.append(ob)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32)}
        return path

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Estimates the advantage function value for each timestep.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        next_values_n = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: next_ob_no})
        q_n = re_n + self.gamma * next_values_n * (1-terminal_n)
        curr_values_n = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: ob_no})
        adv_n = q_n - curr_values_n

        if self.normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                re_n: length: sum_of_path_lengths. Each element in re_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                nothing
        """
        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                next_values_n = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: next_ob_no})
                target_n = re_n + self.gamma * next_values_n * (1 - terminal_n)
            _, loss = self.sess.run([self.critic_update_op, self.critic_loss],
                                    feed_dict={self.sy_ob_no: ob_no, self.sy_target_n: target_n})

    def update_actor(self, ob_no, ac_na, adv_n):
        """ 
            Update the parameters of the policy.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated
                    advantages whose length is the sum of the lengths of the paths

            returns:
                nothing

        """
        self.sess.run(self.actor_update_op,
            feed_dict={self.sy_ob_no: ob_no, self.sy_ac_na: ac_na, self.sy_adv_n: adv_n})


def train_AC(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate,
        num_target_updates,
        num_grad_steps_per_target_update,
        animate, 
        logdir, 
        normalize_advantages,
        seed,
        n_layers,
        size,
        ########################################################################
        # Exploration args
        bonus_coeff,
        kl_weight,
        density_lr,
        density_train_iters,
        density_batch_size,
        density_hiddim,
        dm,
        replay_size,
        sigma,
        ########################################################################
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
    ########################################################################
    # Exploration
    if env_name == 'PointMass-v0':
        from pointmass import PointMass
        env = PointMass()
    else:
        env = gym.make(env_name)
    dirname = logz.G.output_dir
    ########################################################################

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        'num_target_updates': num_target_updates,
        'num_grad_steps_per_target_update': num_grad_steps_per_target_update,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_advantage_args = {
        'gamma': gamma,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_advantage_args) #estimate_return_args

    # build computation graph
    agent.build_computation_graph()

    ########################################################################
    # Initalize exploration density model
    if dm != 'none':
        if env_name == 'PointMass-v0' and dm == 'hist':
            density_model = Histogram(
                nbins=env.grid_size, 
                preprocessor=env.preprocess)
            exploration = DiscreteExploration(
                density_model=density_model,
                bonus_coeff=bonus_coeff)
        elif dm == 'rbf':
            density_model = RBF(sigma=sigma)
            exploration = RBFExploration(
                density_model=density_model,
                bonus_coeff=bonus_coeff,
                replay_size=int(replay_size))
        elif dm == 'ex2':
            density_model = Exemplar(
                ob_dim=ob_dim, 
                hid_dim=density_hiddim,
                learning_rate=density_lr, 
                kl_weight=kl_weight)
            exploration = ExemplarExploration(
                density_model=density_model, 
                bonus_coeff=bonus_coeff, 
                train_iters=density_train_iters, 
                bsize=density_batch_size,
                replay_size=int(replay_size))
            exploration.density_model.build_computation_graph()
        else:
            raise NotImplementedError

    ########################################################################

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    ########################################################################
    if dm != 'none':
        exploration.receive_tf_sess(agent.sess)
    ########################################################################

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = np.concatenate([path["reward"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        terminal_n = np.concatenate([path["terminal"] for path in paths])

        ########################################################################
        # Modify the reward to include exploration bonus
        """
            1. Fit density model
                if dm == 'ex2':
                    the call to exploration.fit_density_model should return ll, kl, elbo
                else:
                    the call to exploration.fit_density_model should return nothing
            2. Modify the re_n with the reward bonus by calling exploration.modify_reward
        """
        old_re_n = re_n
        if dm == 'none':
            pass
        else:
            # 1. Fit density model
            if dm == 'ex2':
                ### PROBLEM 3
                ### YOUR CODE HERE
                raise NotImplementedError
            elif dm == 'hist' or dm == 'rbf':
                ### PROBLEM 1
                ### YOUR CODE HERE
                raise NotImplementedError
            else:
                assert False

            # 2. Modify the reward
            ### PROBLEM 1
            ### YOUR CODE HERE
            raise NotImplementedError

            print('average state', np.mean(ob_no, axis=0))
            print('average action', np.mean(ac_na, axis=0))

            # Logging stuff.
            # Only works for point mass.
            if env_name == 'PointMass-v0':
                np.save(os.path.join(dirname, '{}'.format(itr)), ob_no)
        ########################################################################
        agent.update_critic(ob_no, next_ob_no, re_n, terminal_n)
        adv_n = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        agent.update_actor(ob_no, ac_na, adv_n)

        if n_iter - itr < 10:
            max_reward_path_idx = np.argmax(np.array([path["reward"].sum() for path in paths]))
            print(paths[max_reward_path_idx]['reward'])

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        ########################################################################
        logz.log_tabular("Unmodified Rewards Mean", np.mean(old_re_n))
        logz.log_tabular("Unmodified Rewards Std", np.mean(old_re_n))
        logz.log_tabular("Modified Rewards Mean", np.mean(re_n))
        logz.log_tabular("Modified Rewards Std", np.mean(re_n))
        if dm == 'ex2':
            logz.log_tabular("Log Likelihood Mean", np.mean(ll))
            logz.log_tabular("Log Likelihood Std", np.std(ll))
            logz.log_tabular("KL Divergence Mean", np.mean(kl))
            logz.log_tabular("KL Divergence Std", np.std(kl))
            logz.log_tabular("Negative ELBo", -elbo)
        ########################################################################
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vac')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=32)
    ########################################################################
    parser.add_argument('--bonus_coeff', '-bc', type=float, default=1e-3)
    parser.add_argument('--density_model', type=str, default='hist | rbf | ex2 | none')
    parser.add_argument('--kl_weight', '-kl', type=float, default=1e-2)
    parser.add_argument('--density_lr', '-dlr', type=float, default=5e-3)
    parser.add_argument('--density_train_iters', '-dti', type=int, default=1000)
    parser.add_argument('--density_batch_size', '-db', type=int, default=64)
    parser.add_argument('--density_hiddim', '-dh', type=int, default=32)
    parser.add_argument('--replay_size', '-rs', type=int, default=int(1e6))
    parser.add_argument('--sigma', '-sig', type=float, default=0.2)
    ########################################################################

    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'ac_' + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_AC(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                num_target_updates=args.num_target_updates,
                num_grad_steps_per_target_update=args.num_grad_steps_per_target_update,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                ########################################################################
                bonus_coeff=args.bonus_coeff,
                kl_weight=args.kl_weight,
                density_lr=args.density_lr,
                density_train_iters=args.density_train_iters,
                density_batch_size=args.density_batch_size,
                density_hiddim=args.density_hiddim,
                dm=args.density_model,
                replay_size=args.replay_size,
                sigma=args.sigma
                ########################################################################
                )

        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_AC in the same thread.
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
