# REF: https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np


# 画像は84x84をn_frame連結したものを想定している
class DoubleDqn(nn.Module):
    def __init__(self, n_frame, n_actions):
        super(DoubleDqn, self).__init__()
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

class DoubleDqnAgent():
    def __init__(self, actions, device, n_frame=4,epsilon=1.0):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.n_frame = n_frame
        self.estimate_probs = False
        self.initialized = False
        self._teacher_model = None
        self.optimizer = None
        self.loss_fn = nn.HuberLoss()
        self.prev_action = None
        self.prev_state = None
        self.gamma = 0.99
        self.device = device

    def initialize(self):
        self.model = DoubleDqn(self.n_frame, len(self.actions)).to(self.device)
        self._teacher_model = DoubleDqn(self.n_frame, len(self.actions)).to(self.device)
        self._teacher_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00025)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    def load(cls, actions, model_path, device, n_frame=4, epsilon=0.0001):
        actions = actions
        agent = cls(epsilon=epsilon, actions=actions, device=device)
        agent.model = DoubleDqn(n_frame, len(actions)).to(device)
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
        agent.model.eval()
        agent.initialized = True
        return agent

    def estimate(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array([state])).float().to(self.device)
            return self.model(state_tensor).cpu().numpy()[0]


    def update(self, weights, states, actions, rewards, next_states, dones):
        # Replay Bufferで同様の処理をすでにしているので不要
        # reward clipping
        # rewards = torch.clamp(rewards, -1, 1)
    
        # Double DQN
        with torch.no_grad():
            # 次の状態でのアクション選択
            next_q_values_online = self.model(next_states)
            next_actions = next_q_values_online.argmax(dim=1)
    
            # ターゲットネットワークでの価値評価
            next_q_values_target = self._teacher_model(next_states)
            max_next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
    
        # 現在のQ値の計算
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
        # TD誤差の計算
        td_errors = target_q_values - current_q_values
    
        # 要素ごとの損失（Huber損失を使用）
        loss_elementwise = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
    
        # 補正重みを適用して損失を計算
        loss = torch.mean(weights * loss_elementwise)
    
        # バックプロパゲーションとオプティマイザーの更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        # 優先度の更新
        td_errors_abs = td_errors.detach().abs().cpu().numpy()
    
        return loss.item(), td_errors_abs

    def update_teacher(self):
        self._teacher_model.load_state_dict(self.model.state_dict())

    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(s)
            if self.estimate_probs:
                action = np.random.choice(self.actions,
                                          size=1, p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)

    def play(self, env, episode_count=10, render=True):
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0

            while not done:
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state

                # if render:
                #     env.render()

            else:
                env.lives = 5
                print("{}: Get reward {}.".format(e, episode_reward))
