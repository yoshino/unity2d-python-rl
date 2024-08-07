# REF: https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/q_learning.py
from collections import defaultdict
import numpy as np

from .el_agent import ELAgent

# epsilonを固定すると学習が進まないので、episodeごとに減衰させるように変更した
class QLearningAgent(ELAgent):
    def __init__(self, actions, epsilon=0.5, gamma=0.9, learning_rate=0.1):
        super().__init__(epsilon)
        self.episode_count = 1000
        self.report_interval = 50
        self.save_interval = None
        self.training = True
        self.actions = actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.Q = defaultdict(lambda: [0] * len(actions))
        self.prev_state = None
        self.prev_action = None
        self.final_epsilon = 0.01
        self.epsilon_decay = 0.999

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            if state in self.Q and sum(self.Q[state]) != 0:
                return np.argmax(self.Q[state])
            else:
                return np.random.randint(len(self.actions))

    def learn(self, state, action, reward, done):
        if self.prev_state is not None and self.prev_action is not None:
            gain = reward + self.gamma * max(self.Q[state])
            estimated = self.Q[self.prev_state][self.prev_action]
            self.Q[self.prev_state][self.prev_action] += self.learning_rate * (gain - estimated)
            self.update_epsilon()
        self.prev_state = state
        self.prev_action = action

        if done:
            self.log(reward)

    def update_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon
