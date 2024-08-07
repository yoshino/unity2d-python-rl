# REF: https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/actor_critic.py
from collections import defaultdict
import numpy as np

from .el_agent import ELAgent

class Actor(ELAgent):

    def __init__(self, actions, grid_n=5):
        super().__init__()
        state_n = grid_n * grid_n
        action_n = len(actions)
        self.actions = actions
        self.Q = np.random.uniform(0, 1, state_n * action_n).reshape((state_n, action_n))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, state):
        a = np.random.choice(
            self.actions, 
            1,
            p=self.softmax(self.Q[state])
        )
        return a[0]

class Critic():

    def __init__(self, grid_n=5):
        state_n = grid_n * grid_n
        self.V = np.zeros(state_n)

# gamma=0.9, learning_rate=0.1 -> 学習がうまく進まない
# gamma=0.9, learning_rate=0.2 -> 不安定だけど、learning_rate=0.1よりはマシ
class ActorCriticAgent(ELAgent):

    def __init__(self, actions, epsilon=None, gamma=0.9, learning_rate=0.2):
        super().__init__(epsilon)
        self.episode_count = 3000
        self.report_interval = 50
        self.save_interval = None
        self.training = True
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.actor = Actor(actions=actions)
        self.critic = Critic()
        self.prev_action = None
        self.prev_state = None

    def policy(self, state):
        return self.actor.policy(state)

    def learn(self, state, action, reward, done):
        if self.prev_state is not None and self.prev_action is not None:
            gain = reward + self.gamma * self.critic.V[state]
            estimated = self.critic.V[self.prev_state]
            td = gain - estimated
            self.actor.Q[self.prev_state][self.prev_action] += self.learning_rate * td
            self.critic.V[self.prev_state] += self.learning_rate * td
        self.prev_action = action
        self.prev_state = state

        if done:
            self.log(reward)
