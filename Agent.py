import numpy as np

class Agent(object):
	
	def __init__(self, learning_rate, gamma, n_actions, n_states, epsilon_start, epsilon_end, epsilon_dec_rate):
		self.lr = learning_rate
		self.gamma = gamma
		self.n_actions = n_actions
		self.n_states = n_states
		self.epsilon = epsilon_start
		self.epsilon_min = epsilon_end
		self.epsilon_dec_rate = epsilon_dec_rate
		self.Q = {}

		self.init_Q()
	
	# Q-Table initialized
	def init_Q(self):
		for state in range(self.n_states):
			for action in range(self.n_actions):
				self.Q[(state, action)] = 0.0
	
	# epsilon greedy strategy
	def choose_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.choice([i for i in range(self.n_actions)])
		else:
			actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
			action = np.argmax(actions)

		return action

	def decrement_epsilon(self):
		self.epsilon = self.epsilon * self.epsilon_dec_rate if self.epsilon > self.epsilon_min else self.epsilon_min

		
	def learn(self, state, action, reward, state_):
		actions = np.array([self.Q[(state_, a)] for a in range(self.n_actions)])
		a_max = np.argmax(actions)
		self.Q[(state,action)] += self.lr*(reward +self.gamma*self.Q[(state_,a_max)]-self.Q[(state, action)])

		self.decrement_epsilon()

	def show_policy(self):
		print(self.Q)