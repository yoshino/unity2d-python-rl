# REF: https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/policy_gradient_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from .fn_agent import FNAgent

Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return torch.softmax(self.output_layer(x), dim=-1)

class PolicyGradientAgent(FNAgent):
    # epsilonは0: policyの値を利用するため
    def __init__(self, actions, epsilon=0.0, batch_size=32, buffer_size=256):
        super().__init__(epsilon, actions)
        self.optimizer = None
        self.scaler = StandardScaler()
        self.loss_fn = nn.MSELoss()
        self.loss_history = []
        self.experiences = deque(maxlen=buffer_size)
        self.gamma = 0.99
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.reward_log = []
        self.training_count = 0
        self.training = False
        self.prev_state = None
        self.prev_action = None
        self.step_count = 0
        self.episode_count = 500
        self.report_interval = 50
        self.save_interval = 100

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        scaler_path = model_path.replace('.pth', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, actions, model_path, epsilon=0.0):
        agent = cls(actions=actions, epsilon=epsilon)

        # モデルの構造を定義してから状態辞書をロード
        input_size = 4 
        agent.model = Net(input_size, 10, len(actions))
        agent.model.load_state_dict(torch.load(model_path))

        scaler_path = model_path.replace('.pth', '_scaler.pkl')
        agent.scaler = joblib.load(scaler_path)

        agent.model.eval()
        agent.initialized = True
        return agent

    def initialize(self):
        input_size = 4                  # カートの位置、加速度、ポールの角度、ポールの倒れる速度
        output_size = len(self.actions) # 左か右かの2択
        self.model = Net(input_size, 10, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        states = np.vstack([e.s for e in self.experiences])
        self.scaler.fit(states)

        self.initialized = True
        print("Done initialization. From now, begin training!")

    def estimate(self, s):
        s = np.array(s).reshape((1, -1))

        # 入力データを標準化
        normalized = self.scaler.transform(s)
        normalized = torch.tensor(normalized, dtype=torch.float32)

        # モデルを評価モードに設定
        self.model.eval()

        # PyTorchでは自動で勾配計算を行わないようにする
        with torch.no_grad():
            action_probs = self.model(normalized).numpy()[0]

        return action_probs

    def learn(self, state, action, reward, done):
        self.step_count += 1
        if self.prev_state is not None or self.prev_action is not None:
            # Experience = namedtuple("Experience",
            #                         ["s", "a", "r", "n_s", "d"])
            self.experiences.append(Experience(self.prev_state, self.prev_action, reward, state, done))

        self.prev_state = state
        self.prev_action = action

        if done:
            # 現在のエピソードの報酬を取得
            rewards = [e.r for e in self.get_recent(self.step_count)]
            self.log(sum(rewards))

            if len(self.experiences) == self.buffer_size and not self.training:
                self.training = True
                self.initialize()
        
            if self.training:
                self.update()

            self.step_count = 0

            if self.initialized:
                self.experiences = []

    def update(self):
        states, actions, rewards = self.make_batch()

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        self.optimizer.zero_grad()

        # policyをもとに選択した行動から得られる報酬を最大化するように学習
        # 実際にエージェントが取った行動の確率をaction_probsから選択します
        action_probs = self.model(states)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ポリシーグラディエント法では、選択された行動の確率の対数に対して負の符号を使用し、それに報酬を乗算したものを最小化する
        loss = -torch.log(selected_probs) * rewards
        loss = loss.mean()
        loss.backward()
        self.loss_history.append(loss.item())
        self.optimizer.step()

        self.training_count += 1

    def make_batch(self):
        recent_rewards = [e.r for e in self.get_recent(self.step_count)]
        policy_experiences = []
        for t, e in enumerate(self.experiences):
            s, a, r, n_s, d = e
            d_r = [_r * (self.gamma ** i) for i, _r in
                   enumerate(recent_rewards[t:])]
            d_r = sum(d_r)
            d_e = Experience(s, a, d_r, n_s, d)
            policy_experiences.append(d_e)

        length = min(self.batch_size, len(policy_experiences))
        batch = random.sample(policy_experiences, length)
        states = np.vstack([e.s for e in batch])
        actions = [e.a for e in batch]
        rewards = [e.r for e in batch]
        scaler = StandardScaler()
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = scaler.fit_transform(rewards).flatten()

        return states, actions, rewards

    def get_recent(self, count):
        try:
            if count > len(self.experiences):
                count = len(self.experiences)
            recent = range(len(self.experiences) - count, len(self.experiences))
            return [self.experiences[i] for i in recent]
        except IndexError:
            print('IndexError')
            print('len(self.experiences): ', len(self.experiences))
            print('count: ', count)
            return []

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(episode, mean, std))
