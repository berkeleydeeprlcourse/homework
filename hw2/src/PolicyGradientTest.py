import unittest
import tensorflow as tf
import numpy as np
from train_pg_f18 import createEnvAndAgent

class PolicyGradientTest(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		self.env, self.agent = createEnvAndAgent(
			exp_name = 'vpg',
			env_name = 'CartPole-v0',
			n_iter = 20,
			gamma = 1.0,
			min_timesteps_per_batch = 10, # batch size
			max_path_length = -1., #ep_len
			learning_rate = 0.001,
			reward_to_go = True,
			animate = False, 
	        normalize_advantages = False,
	        nn_baseline = False, 
	        seed = 1234,
	        n_layers = 1,
	        size = 5)
		self.agent.init_tf_sess()

	def test_get_log_prob_discrete(self):
		# policy params: sy_logits_na: (batch_size, self.ac_dim)
		# sy_ac_na: (batch_size,)
		agent = self.agent
		batch_size = 10
		policyParams = tf.placeholder(shape=[None, agent.ac_dim], name="policy", dtype=tf.float32)
		sy_ac_na = tf.placeholder(shape=[None], name = "actions", dtype=tf.int32)
		policyParams_input = [generate_softmax_distribution(agent.ac_dim) for i in range(batch_size)]
		action_input = [np.random.randint(agent.ac_dim) for i in range(batch_size)]
		expected_output = [policyParams_input[i][action_input[i]] for i in range(batch_size)]

		sample_action_prob = agent.get_log_prob(policyParams, sy_ac_na)


		print(policyParams_input)
		print(action_input)
		print(expected_output)

		output = agent.sess.run(sample_action_prob, feed_dict = {
			policyParams:policyParams_input,
			sy_ac_na: action_input})

		print(output)
		self.assertTrue(agent.discrete)
		self.assertTrue(output == expected_output)
		

	# def test_reward(self):
	# 	(env, agent) = createEnvAndAgent(
	# 		exp_name = 'vpg',
	#         env_name = 'CartPole-v0',
	#         n_iter = 20, 
	#         gamma = 1.0, 
	#         min_timesteps_per_batch = 10, # batch size 
	#         max_path_length = -1., #ep_len
	#         learning_rate = 0.001, 
	#         reward_to_go = True, 
	#         animate = False, 
	#         normalize_advantages = False,
	#         nn_baseline = False, 
	#         seed = 1234,
	#         n_layers = 1,
	#         size = 5)
	# 	self.assertEqual('foo'.upper(), 'FOO')

	# def test_advantage_norm(self):
	# 	(env, agent) = createEnvAndAgent(
	# 		exp_name = 'vpg',
	#         env_name = 'CartPole-v0',
	#         n_iter = 20, 
	#         gamma = 1.0, 
	#         min_timesteps_per_batch = 10, # batch size 
	#         max_path_length = -1., #ep_len
	#         learning_rate = 0.001, 
	#         reward_to_go = True, 
	#         animate = False, 
	#         normalize_advantages = False,
	#         nn_baseline = False, 
	#         seed = 1234,
	#         n_layers = 1,
	#         size = 5)
	# 	self.assertEqual('foo'.upper(), 'FOO')

def generate_softmax_distribution(num_labels):
	output = np.zeros(num_labels)
	total = 1
	if (num_labels == 1):
		return np.array(1.0)
	else:
		for i in range(num_labels):
			random_prob = np.random.rand()*0.5
			if (total >= random_prob):
				output[i] = random_prob
				total -= random_prob
			elif total == 0:
				output[i] = 0
			else:
				output[i] = total
				total = 0
		return output

if __name__ == '__main__':
    unittest.main()