# REF: https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/value_function_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np

from .fn_agent import FNAgent

Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class ValueFunctionAgent(FNAgent):
    def __init__(self, actions, epsilon=1.0, batch_size=32, buffer_size=1024):
        super().__init__(epsilon, actions)
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        self.loss_history = []
        self.prev_action = None
        self.prev_state = None
        self.experiences = deque(maxlen=buffer_size)
        self.gamma = 0.99
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.reward_log = []
        self.training = False
        self.final_epsilon = 0.01
        self.epsilon_decay = 0.9999
        self.episode_count = 1000
        self.save_interval = 100
        self.report_interval = 50
        self.episode_rewards = []

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    def load(cls, actions, model_path, epsilon=0.0001):
        agent = cls(actions=actions, epsilon=epsilon)

        # モデルの構造を定義してから状態辞書をロード
        input_size = 4 
        agent.model = Net(input_size, 10, len(actions))
        agent.model.load_state_dict(torch.load(model_path))

        agent.model.eval()
        agent.initialized = True
        return agent

    def initialize(self):
        input_size = 4                  # カートの位置、加速度、ポールの角度、ポールの倒れる速度
        output_size = len(self.actions) # 左か右かの2択
        self.model = Net(input_size, 10, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def estimate(self, s):
        s = np.array(s, dtype=np.float32)
        with torch.no_grad():
            estimated = self.model(torch.from_numpy(s).float()).numpy()
        return estimated

    def learn(self, state, action, reward, done):
        if self.prev_state is not None or self.prev_action is not None:
            # Experience = namedtuple("Experience",
            #                         ["s", "a", "r", "n_s", "d"])
            self.experiences.append(Experience(self.prev_state, self.prev_action, reward, state, done))

            # trainingを開始するタイミング
            if len(self.experiences) == self.buffer_size and not self.training:
                self.training = True
                self.initialize()

            # training中の処理
            if self.training:
                batch = random.sample(self.experiences, self.batch_size)
                self.update(batch)
                self.update_epsilon()

        self.episode_rewards.append(reward)
        self.prev_state = state
        self.prev_action = action

        if done:
            reward_sum = sum(self.episode_rewards)
            self.log(reward_sum)

            self.prev_state = None
            self.prev_action = None
            self.episode_rewards = []

    def update(self, batch_experiences):
        states = np.vstack([e.s for e in batch_experiences]).astype(np.float32)
        n_states = np.vstack([e.n_s for e in batch_experiences]).astype(np.float32)

        estimateds = self.estimate(states)
        future = self.estimate(n_states)

        rewards = np.array([e.r for e in batch_experiences], dtype=np.float32)
        actions = np.array([e.a for e in batch_experiences], dtype=np.int64)
        dones = np.array([e.d for e in batch_experiences], dtype=np.bool)

        # targets: 教師信号
        # state, actionにおける期待報酬を計算（正解ラベル)
        targets = estimateds.copy()
        for idx in range(len(batch_experiences)):
            targets[idx, actions[idx]] = rewards[idx] + self.gamma * np.max(future[idx]) * (not dones[idx])

        self.optimizer.zero_grad()
        outputs = self.model(torch.from_numpy(states).float())
        loss = self.loss_fn(outputs, torch.from_numpy(targets).float())
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())

    def update_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon
            # print(f"Updated epsilon: {self.epsilon}")

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}). Experience length is {}".format(
                   episode, mean, std, len(self.experiences)))
            print("experience length: ", len(self.experiences))
            self.reward_log = []
