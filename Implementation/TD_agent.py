from TD_Gammon2 import TDGammon
import numpy as np

class TDAgent:
	def __init__(self,num_actions, num_inputs, num_hidden, num_output, lr=0.1, d=0.95, l=0.7):
		self.num_actions = num_actions
		self.num_inputs = num_inputs
		self.tdNet = TDGammon(num_inputs, num_hidden, num_output, lr, d, l)
		self.input_layer = []
	
	def choose_action(self, observation):
		#if np.random.uniform()>0.9:
		#	return np.random.choice(range(self.num_actions))
		state_values = []
		#max_state_value = 0
		#max_action = 0
		for i in range(self.num_actions):
			new_state = self.tdNet.model(self.model_func,observation,i)
			state_value = self.tdNet.getValue(self.state_to_input(new_state))
			#if max_state_value < state_value[0]:
			#	max_state_value = state_value[0]
			#	max_action = i
			state_values.append(state_value[0])
		print(state_values)
		enum_state_values = np.random.permutation(list(enumerate(state_values)))
		action = max(enum_state_values, key=lambda t:t[1])[0]
		
		return int(action)
		#return max_action
		
	def state_to_input(self, state):
		inputs = [0 for _ in range(self.num_inputs)]
		#if state > 14:
		#	inputs[15] = 1
		#else:
		#	inputs[state] = 1
		inputs[state] = 1
		return inputs
	
	def learn(self, s, r):
		self.tdNet.input_layer = self.state_to_input(s)
		self.tdNet.feedforward()
		self.tdNet.TDlearn(r,0)
		
	def model_func(self,current_state, action):
		
		if action == 0:
			if current_state > 5:
				current_state -= 5
		elif action == 1:
			if current_state < 15:
				current_state += 5
		elif action == 2:
			if current_state % 5 != 0:
				current_state -= 1
		elif action == 3:
			if current_state % 5 != 4:
				current_state += 1
				
		return current_state