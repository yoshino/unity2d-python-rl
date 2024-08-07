# REF: https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/a2c_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from collections import deque, namedtuple
import numpy as np
import joblib

from .fn_agent import FNAgent

Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])
class SampleLayer(nn.Module):
    def __init__(self):
        super(SampleLayer, self).__init__()
        self.output_dim = 1  # 評価から1つのアクションをサンプリング

    def forward(self, x):
        noise = torch.rand_like(x)
        return torch.argmax(x - torch.log(-torch.log(noise)), dim=1)


class ActorCriticNet(nn.Module):
    def __init__(self, feature_n, output_n):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(feature_n, 10)
        self.fc2 = nn.Linear(10, 10)
        self.actor = nn.Linear(10, output_n)
        self.critic = nn.Linear(10, 1)
        self.sample_layer = SampleLayer()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Actor
        # 全結合層で計算した値(Q(s,a))からSampleLayerでアクションをサンプリング
        action_evals = self.actor(x)
        actions = self.sample_layer(action_evals)
        # Critic
        # 全結合層で計算した値(Q(s,a))から状態価値を計算
        values = self.critic(x)
        return actions, action_evals, values

# 画像は84x84をn_frame連結したものを想定している
class ActorCriticConvNet(nn.Module):
    def __init__(self, frame_n, output_n):
        super(ActorCriticConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=frame_n, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)

        self.actor = nn.Linear(256, output_n)
        self.critic = nn.Linear(256, 1)
        self.sample_layer = SampleLayer()

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.actor.weight)
        torch.nn.init.kaiming_normal_(self.critic.weight)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        # Actor
        # 全結合層で計算した値(Q(s,a))からSampleLayerでアクションをサンプリング
        action_evals = self.actor(x)
        actions = self.sample_layer(action_evals)
        # Critic
        # 全結合層で計算した値(Q(s,a))から状態価値を計算
        values = self.critic(x)
        return actions, action_evals, values

class AdvantageActorCriticAgent(FNAgent):
    # buffer_size=256だと学習がうまく行かなかった
    # episodeは4200回くらいから学習が安定し出した
    def __init__(self, actions, device, epsilon=0.0, batch_size=32, buffer_size=512):
        # ActorCriticAgent uses self policy (doesn't use epsilon).
        super().__init__(epsilon=epsilon, actions=actions)
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
        self.episode_count = 5000
        self.report_interval = 100
        self.save_interval = 500
        self.device = device

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        scaler_path = model_path.replace('.pth', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(cls, actions, model_path, device):
        agent = cls(actions)
        agent.model = ActorCriticConvNet(frame_n=4, output_n=len(actions)).to(device)
        agent.model.load_state_dict(torch.load(model_path, map_location=device))

        scaler_path = model_path.replace('.pth', '_scaler.pkl')
        agent.scaler = joblib.load(scaler_path)

        agent.model.eval()
        agent.initialized = True
        return agent

    def initialize(self):
        self.model = ActorCriticConvNet(frame_n=4, output_n=len(self.actions)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def policy(self, state):
        if not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            state_tensor = torch.from_numpy(np.array([state])).float().to(self.device)
            with torch.no_grad():
                action, action_evals, values = self.model(state_tensor)
                action = action.cpu().numpy()[0]

            return action

    # 状態価値を返す
    def estimate(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array([state])).float().to(self.device)
            action, action_evals, values = self.model(state_tensor)
            values = values.cpu().numpy()[0][0]

        return values

    def learn(self, state, action, reward, done):
        if self.prev_state is not None and self.prev_action is not None:
            # Experience = namedtuple("Experience",
            #                         ["s", "a", "r", "n_s", "d"])
            self.experiences.append(Experience(self.prev_state, self.prev_action, reward, state, done))

            if len(self.experiences) == self.buffer_size and not self.training:
                self.training = True
                self.initialize()
            
            if done:
                reward_sum = sum([e.r for e in self.experiences])
                self.log(reward_sum)

            if len(self.experiences) == self.buffer_size and self.training:
                self.update()
                self.experiences.clear()

        self.prev_state = state
        self.prev_action = action

    def update(self):
        states, actions, values = self.make_batch()

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)  
        values = torch.tensor(values, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()

        # actor, criticは層を共有して異なる出力を返す
        # actor ->  action_evals(行動評価) -> predicted_actions(行動確率, これは計算に用いない)
        # critic -> estimated_values
        predicted_actions, action_evals, estimated_values = self.model(states)

        # -log_π(a|s)
        neg_log_prob = F.cross_entropy(action_evals, actions, reduction='none')
        # advantages(符号は逆->最小化): 状態における行動価値(estimated_values) - 状態価値(values)
        # 純粋な行動の価値を算出する
        advantages = values - estimated_values.detach() # detach(policy_loss)で勾配計算から外す(Actor側からの勾配更新()がCritic側に適用されるのを防ぐため)
        # actor側の目的関数: advantageの期待値(符号は逆なので最小化)
        policy_loss = torch.mean(neg_log_prob * advantages)

        # critic側の目的関数
        value_loss = self.loss_fn(estimated_values.squeeze(), values)

        # actionが１つの行動に隔たらないようにするため
        entropy = torch.mean(self.categorical_entropy(action_evals))

        # actor側の目的関数とcritic側の目的関数を同時に更新するために、足し合わせる
        value_loss_weight=1.0
        entropy_weight=0.1
        total_loss = policy_loss + value_loss_weight * value_loss
        total_loss -= entropy_weight * entropy

        # 逆伝播と最適化
        total_loss.backward()
        self.optimizer.step()

    def make_batch(self):
        values = []

        states = torch.stack([e.s for e in self.experiences]).to(self.device)
        actions = np.array([e.a for e in self.experiences])

        # Calculate values.
        # If the last experience isn't terminal (done) then estimates value.
        last = self.experiences[-1]
        future = last.r if last.d else self.estimate(last.n_s)
        for e in reversed(self.experiences):
            value = e.r
            if not e.d:
                value += self.gamma * future
            values.append(value)
            future = value

        values = np.array(list(reversed(values)))
        values = self.scaler.fit_transform(values.reshape((-1, 1))).flatten()

        return states.cpu().numpy(), actions, values

    def categorical_entropy(self, logits):
        prob = F.softmax(logits, dim=1)
        log_prob = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(prob * log_prob, dim=1)
        return entropy

    def show_reward_log(self, interval=3, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(episode, mean, std))

class AdvantageActorCriticNetAgent(AdvantageActorCriticAgent):
    def __init__(self, actions, device="cpu", epsilon=0.0, batch_size=32, buffer_size=256):
        super().__init__(epsilon=epsilon, actions=actions, device=device)
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
        self.episode_count = 2000
        self.report_interval = 100
        self.save_interval = 500

    def initialize(self):
        feature_shape = self.experiences[0].s.shape
        self.model = ActorCriticNet(feature_shape[0], len(self.actions))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    @classmethod
    def load(cls, actions, model_path):
        agent = cls(actions)
        agent.model = ActorCriticNet(feature_n=4, output_n=len(actions))
        agent.model.load_state_dict(torch.load(model_path))

        scaler_path = model_path.replace('.pth', '_scaler.pkl')
        agent.scaler = joblib.load(scaler_path)

        agent.model.eval()
        agent.initialized = True
        return agent

    def policy(self, s):
        if not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            s = np.array(s).reshape((1, -1))
            s = torch.tensor(s, dtype=torch.float32)

            with torch.no_grad():
                action, action_evals, values = self.model(s)
                action = action.numpy()[0]

            return action

    def estimate(self, s):
        s = np.array(s).reshape((1, -1))
        s = torch.tensor(s, dtype=torch.float32)

        with torch.no_grad():
            action, action_evals, values = self.model(s)
            values = values.numpy()[0][0]

        return values

    def make_batch(self):
        values = []
        states = np.array([e.s for e in self.experiences])
        actions = np.array([e.a for e in self.experiences])

        # Calculate values.
        # If the last experience isn't terminal (done) then estimates value.
        last = self.experiences[-1]
        future = last.r if last.d else self.estimate(last.n_s)
        for e in reversed(self.experiences):
            value = e.r
            if not e.d:
                value += self.gamma * future
            values.append(value)
            future = value

        values = np.array(list(reversed(values)))
        values = self.scaler.fit_transform(values.reshape((-1, 1))).flatten()

        return states, actions, values
