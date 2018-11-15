import unittest

class PolicyGradientTest(unittest.TestCase):

	def test_sample_action(self):
		(env, agent) = createEnvAndAgent(
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

		sample_action = agent.sample_action()
		self.assertEqual('foo'.upper(), 'FOO')

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

if __name__ == '__main__':
	unittest.main()