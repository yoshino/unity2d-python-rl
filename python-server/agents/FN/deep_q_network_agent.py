# REF: https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/dqn_agent.py
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

# 画像は84x84をn_frame連結したものを想定している
class DeepQNetwork(nn.Module):
    def __init__(self, n_frame, n_actions):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_frame, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, n_actions)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepQNetworkAgent(FNAgent):
    def __init__(self, actions, device, decay=1000, epsilon=1.0, batch_size=32, buffer_size=10000):
        super().__init__(epsilon, actions)
        self._teacher_model = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        self.loss_history = []
        self.prev_action = None
        self.prev_state = None
        self.experiences = deque(maxlen=buffer_size)
        self.gamma = 0.99
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.initial_epsilon = epsilon
        self.final_epsilon = 0.01
        self.teacher_update_freq = 3
        self.reward_log = []
        self.training_count = 0
        self.decay = decay
        self.training = False
        self.episode_count = 1000
        self.save_interval = 100
        self.report_interval = 50
        self.device = device

    def initialize(self):
        self.model = DeepQNetwork(4, len(self.actions)).to(self.device)
        self._teacher_model = DeepQNetwork(4, len(self.actions)).to(self.device)
        self._teacher_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    def load(cls, actions, model_path, epsilon=0.0001):
        actions = actions
        agent = cls(epsilon=epsilon, actions=actions)
        agent.model = DeepQNetwork(4, len(actions)).to(self.device)
        agent.model.load_state_dict(torch.load(model_path, map_location=self.device))
        agent.model.eval()
        agent.initialized = True
        return agent

    def estimate(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array([state])).float().to(self.device)
            return self.model(state_tensor).cpu().numpy()[0]

    def learn(self, state, action, reward, done):
        if self.prev_state is not None and self.prev_action is not None:
            self.experiences.append(Experience(self.prev_state, self.prev_action, reward, state, False))

            # trainingを開始するタイミング
            if len(self.experiences) == self.buffer_size and not self.training:
                self.training = True
                self.initialize()

            # training中の処理
            if len(self.experiences) >= self.buffer_size:
                batch = random.sample(self.experiences, self.batch_size)
                self.update(batch)
                self.training_count += 1

                if self.training_count % self.teacher_update_freq == 0:
                    self.update_teacher()

        self.prev_state = state
        self.prev_action = action

        diff = (self.initial_epsilon - self.final_epsilon)
        decay = diff / self.decay
        self.epsilon = max(self.epsilon - decay, self.final_epsilon)

    def update(self, batch_experiences):
        # 1. 状態と次の状態の準備
        states = torch.stack([e.s for e in batch_experiences]).to(self.device)
        n_states = torch.stack([e.n_s for e in batch_experiences]).to(self.device)

        # 2. 遷移先の価値の計算
        estimateds = self.model(states)
        with torch.no_grad():
            future = self._teacher_model(n_states)

        # 3. 報酬の更新
        updated_estimateds = estimateds.clone()
        for i, e in enumerate(batch_experiences):
            reward = e.r
            if not e.d:
                reward += self.gamma * torch.max(future[i]).item()  # .item()を追加してスカラーに変換
            updated_estimateds[i][e.a] = reward

        # 4. 損失の計算とバックプロパゲーション
        loss = self.loss_fn(updated_estimateds, estimateds)
        self.optimizer.zero_grad()
        loss.backward()

        # 5. 勾配のクリッピングとオプティマイザーのステップ
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.loss_history.append(loss.item())

    def update_teacher(self):
        self._teacher_model.load_state_dict(self.model.state_dict())
