import pandas as pd
import random
import math
import numpy as np


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

	def reset(self):
		self.finish = False
		self.player_pos = 15
		self.step_log = []
		self.path_log = []

	def run_game(self, agent, n_games):

		for _ in range(n_games):
			reward_accumulate = 0
			reward = 0
			while not self.finish:
				action_id = agent.choose_action(self.player_pos)
				action = possible_actions[action_id]
				self.path_log.append(self.player_pos)
				agent.previous_state = self.player_pos
				agent.previous_action = action_id
				#print()
				#print(action_id,action == UP)
				if self.player_pos <=15:
					reward = -1
				elif self.player_pos == 19:
					reward = 100
					self.finish = True
				elif self.player_pos >= 16 and self.player_pos <= 18:
					reward =- 100
					self.finish = True
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

				reward_accumulate += reward

				if not self.finish:
					self.step_log.append(action)
					if agent.previous_state is not None:
						#print(agent.previous_action)
						#print(sdkjfnk)
						agent.qlearn.learn(agent.previous_state,agent.previous_action,reward,self.player_pos)
				else:
					agent.qlearn.learn(agent.previous_state,agent.previous_action,reward_accumulate,100)

			print('reward',reward_accumulate)
			agent.scores = agent.scores.append(pd.DataFrame([reward_accumulate], columns = agent.scores.columns), ignore_index = True)

			self.reset()
		agent.scores.to_pickle(self.rname, 'gzip')

class QLAgent:
	def __init__(self, lr, df, eps):
		self.qlearn = QLearningTable(actions = list(range(len(possible_actions))),learning_rate=lr, reward_decay=df, e_greedy=eps)
		self.previous_action = None
		self.previous_state = None
		self.scores = pd.DataFrame(columns=['score'], dtype=np.float64)

	def choose_action(self, observation):
		return self.qlearn.choose_action(observation)


class RandomAgent:

	def choose_action(self):

		return random.randint(0,3)

class QLearningTable:

	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions)

	def choose_action(self, observation):
		self.check_state_exist(observation)

		if np.random.uniform()<self.epsilon:
			#choose best action
			state_action = self.q_table.ix[observation,:]

			#some actions have same value
			state_action = state_action.reindex(np.random.permutation(state_action.index))

			action = state_action.argmax()
		else:
			#choose random action
			#useful_actions = [item for item in self.actions if item not in useless_actions]
			action = np.random.choice(self.actions)

		return action

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		self.check_state_exist(s)

		q_predict = self.q_table.ix[s,a]

		if s_ != 100:
			q_target = r + self.gamma * self.q_table.ix[s_,:].max()
			#print('target:',self.q_table.ix[s_,:])
		else:
			q_target = r

		#update
		self.q_table.ix[s,a] += self.lr * (q_target - q_predict)

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			#append new state to q table
			self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index = self.q_table.columns, name = state))

if __name__ == '__main__':
	
	cont1 = 'lr_'
	cont2 = '_df_'
	cont3 = '_eps_'
	
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
0.9,
0.8,
0.6,
0.4,
0.2,
0.1,
0.01,
0.001,
0,
	]
	
	e_gs = [1,
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
	
	for i in e_gs:
		lr = 0.01
		df = 0.9
		eps = i
		rname = cont1 + ('%.4f' % lr) + cont2 + ('%.4f' % df) + cont3 + ('%.4f' % eps) + '.gz'
		game = SimpleGame(rname)
		agent = QLAgent(lr,df,eps)
		game.run_game(agent, 1000)
	#for j in range(1,11):
	#	lr = 0.01
	#	df = 1.0 - 0.02 * j
	#	eps = 0.9
	#	print(lr, df)
	#game = SimpleGame()
	#agent = QLAgent()
	#game.run_game(agent, 2000)
	#print(agent.qlearn.q_table)
	#agent.qlearn.q_table.to_pickle('simple_q_table.gz', 'gzip')