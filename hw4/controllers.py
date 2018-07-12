import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		self.env = env

	def get_action(self, state):
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		obs, obs_list, obs_next_list, act_list = [], [], [], []
        
		[obs.append(state) for _ in range(self.num_simulated_paths)]
		
		for _ in range(self.horizon):
			obs_list.append(obs)

			# get random actions
			actions = []
			[actions.append(self.env.action_space.sample()) for _ in range(self.num_simulated_paths)]
			
			act_list.append(actions)
			obs = self.dyn_model.predict(np.array(obs), np.array(actions))
			obs_next_list.append(obs)

		trajectory_cost_list = trajectory_cost_fn(self.cost_fn, np.array(obs_list), np.array(act_list), np.array(obs_next_list)) 
        
		j = np.argmin(trajectory_cost_list)
        
		return act_list[0][j]

