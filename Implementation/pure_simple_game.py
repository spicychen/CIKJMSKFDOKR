import pandas as pd
import random
import math
import numpy as np
from TD_agent import TDAgent
import matplotlib.pyplot as plt

#directions
UP = 'U'
DOWN = 'D'
LEFT = 'L'
RIGHT = 'R'

#actions
possible_actions = [
	UP,
	DOWN,
	LEFT,
	RIGHT
	]


class SimpleGame:

	def __init__(self,result_name):
		super(SimpleGame,self).__init__()
		self.finish = False
		self.player_pos = 15
		self.step_log = []
		self.path_log = []
		self.rname = result_name
		self.scores = pd.DataFrame(columns=['score'], dtype=np.float64)
		
	def reset(self):
		self.finish = False
		self.player_pos = 15
		self.step_log = []
		self.path_log = []

	def run_game(self, agent, n_games):

		for _ in range(n_games):
			self.reset()
			reward_accumulate = 0
			reward = 0
			while not self.finish:
				action_id = agent.choose_action(self.player_pos)
				action = possible_actions[action_id]
				self.path_log.append(self.player_pos)
				pre_pos = self.player_pos
				
				if action == UP:
					if self.player_pos > 5:
						self.player_pos -= 5
				elif action == DOWN:
					if self.player_pos < 15:
						self.player_pos += 5
				elif action == LEFT:
					if self.player_pos % 5 != 0:
						self.player_pos -= 1
				elif action == RIGHT:
					if self.player_pos % 5 != 4:
						self.player_pos += 1
				if self.player_pos <=15:
					reward = 0
				elif self.player_pos == 19:
					reward = 1
					self.finish = True
				elif self.player_pos >= 16 and self.player_pos <= 18:
					reward = -1
					self.finish = True

				reward_accumulate += reward
				#print(pre_pos,reward)
				agent.learn(self.player_pos,reward)
				
				self.step_log.append(action)
				#print(self.step_log)
					
			print('reward',reward_accumulate)
			self.scores = self.scores.append(pd.DataFrame([reward_accumulate], columns = self.scores.columns), ignore_index = True)
			
			
		self.scores.to_pickle(self.rname + '.gz', 'gzip')
		
if __name__ == '__main__':
	
	
	cont0 = 'TD result/'
	cont1 = 'lr_'
	cont2 = '_df_'
	cont3 = '_etdf_'
	cont4 = '_nhid_'
	
	lrs = [0,
0.0001,
0.001,
0.01,
0.05,
0.1,
0.2,
0.5,
0.8,
1,
	]
	dfs = [1,
0.95,
0.9,
0.8,
0.6,
0.4,
0.2,
0.1,
0.01,
0,
	]
	ets = [1,
0.9,
0.8,
0.7,
0.6,
0.5,
0.4,
0.3,
0.2,
0.1,
	]
	num_of_hid=[20,
18,
16,
14,
12,
10,
8,
6,
4,
2,
	]
	
	#for i in num_of_hid[9:]:
		
	#lr = 0.1
	#df = 0.95
	#lam = 0.7
	#nhid = i
	
	#rname = cont0+cont1+('%.4f' % lr) + cont2 + ('%.4f' % df) + cont3 + ('%.4f' % lam) + cont4 + ('%d' % nhid)
	game = SimpleGame('tdGammonR')
	agent = TDAgent(len(possible_actions),20,5,1)
	game.run_game(agent,10000)
	
#	test_game = SimpleGame('test')
#	test_game.run_game(agent,1)
#	print(test_game.step_log)

	a = agent.tdNet.input_weights
	b = agent.tdNet.hidden_weights
	print(a)
	print(b)
	plt.subplot(121)
	plt.xticks(np.array([0,1,2,3,4]))
	plt.yticks(np.array(range(20)))
	plt.imshow(np.array(a))
	plt.subplot(122)
	plt.imshow(np.array(b))
	plt.show()
	#"""