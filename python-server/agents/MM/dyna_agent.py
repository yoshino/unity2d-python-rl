# REF: https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/MM/dyna.py
import numpy as np
from collections import defaultdict, Counter


class Model():
    def __init__(self, actions):
        self.num_actions = len(actions)
        self.transit_count = defaultdict(lambda: [Counter() for a in actions])
        self.total_reward = defaultdict(lambda: [0] *
                                        self.num_actions)
        self.history = defaultdict(Counter)

    # 実環境の遷移回数や報酬の合計を更新
    def update(self, state, action, reward, next_state):
        self.transit_count[state][action][next_state] += 1
        self.total_reward[state][action] += reward
        self.history[state][action] += 1

    # 状態と行動から次状態をサンプリング
    def transit(self, state, action):
        counter = self.transit_count[state][action]
        states = []
        counts = []
        for s, c in counter.most_common():
            states.append(s)
            counts.append(c)
        probs = np.array(counts) / sum(counts)
        return np.random.choice(states, p=probs)

    # 状態と行動から報酬をサンプリング
    def reward(self, state, action):
        total_reward = self.total_reward[state][action]
        total_count = self.history[state][action]
        return total_reward / total_count

    def simulate(self, count):
        states = list(self.transit_count.keys())
        actions = lambda s: [a for a, c in self.history[s].most_common()
                             if c > 0]

        for i in range(count):
            state = np.random.choice(states)
            action = np.random.choice(actions(state))

            next_state = self.transit(state, action)
            reward = self.reward(state, action)

            yield state, action, reward, next_state

# epsilonが大きい(0.1)とrewardが安定しない
class DynaAgent():
    def __init__(self, actions, steps_in_model=1, epsilon=0.01):
        self.epsilon = epsilon
        self.actions = actions
        self.steps_in_model = steps_in_model
        self.reward_log = []
        self.value = defaultdict(lambda: [0] * len(actions))
        self.initialized = False
        self.training = False
        self.gamma = 0.9
        self.learning_rate = 0.1
        self.prev_state = None
        self.prev_action = None
        self.report_interval = 100
        self.save_interval = None
        self.episode_count = 5000

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            if sum(self.value[state]) == 0:
                return np.random.randint(len(self.actions))
            else:            
                return np.argmax(self.value[state])

    def learn(self, state, action, reward, done):
        if not self.initialized:
            self.model= Model(self.actions)
            self.initialized = True
            self.training = True

        if self.prev_state is not None and self.prev_action is not None:
            gain = reward + self.gamma * max(self.value[state])
            estimated = self.value[self.prev_state][self.prev_action]
            self.value[self.prev_state][self.prev_action] += self.learning_rate * (gain - estimated)

            if self.steps_in_model > 0:
                self.model.update(self.prev_state, self.prev_action, reward, state)
                for s, a, r, n_s in self.model.simulate(self.steps_in_model):
                    gain = r + self.gamma * max(self.value[n_s])
                    estimated = self.value[s][a]
                    self.value[s][a] += self.learning_rate * (gain - estimated)

        self.prev_state = state
        self.prev_action = action

        if done:
            self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(episode, mean, std))
            print("Current epsilon: {}".format(self.epsilon))
