# REF: https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/sarsa.py
from collections import defaultdict
import numpy as np

from .el_agent import ELAgent

class SarsaAgent(ELAgent):

    def __init__(self, actions, epsilon=0.1, gamma=0.9, learning_rate=0.1):
        super().__init__(epsilon)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.Q = defaultdict(lambda: [0] * len(actions))
        self.actions = actions
        self.prev_action = None
        self.prev_state = None
        self.report_interval = 100
        self.save_interval = None
        self.episode_count = 2000

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
            gain = reward + self.gamma * self.Q[state][action]
            estimated = self.Q[self.prev_state][self.prev_action]
            self.Q[self.prev_state][self.prev_action] += self.learning_rate * (gain - estimated)

        self.prev_state = state
        self.prev_action = action

        if done:
            self.log(reward)

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(episode, mean, std))
