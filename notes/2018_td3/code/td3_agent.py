import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class ActorNetwork(nn.Module):
    ACTION_RANGE = 2.0

    def __init__(self, state_dim, action_dim, device):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # ネットワークサイズは参考コードに近い値に拡大
        # 例: hidden1=400, hidden2=300
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.to(device)
        self._initialize_weights()

    def forward(self, s):
        s = s.to(self.device)
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        # 出力をtanhで[-1,1]にクリップし、その後ACTION_RANGEをかける
        actions = torch.tanh(self.fc3(x)) * self.ACTION_RANGE
        return actions

    def _initialize_weights(self):
          """Kaiming Normalで初期化"""
          for m in self.modules():
              if isinstance(m, nn.Linear):
                  torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # ReLUに適した初期化
                  if m.bias is not None:
                      torch.nn.init.constant_(m.bias, 0)  # バイアスはゼロで初期化
  
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        # 1. Clipped Double Q learning
        # -> 2つのCriticネットワークの出力のうち小さい方を選択
        self.fc1_2 = nn.Linear(state_dim + action_dim, 64)
        self.fc2_2 = nn.Linear(64, 64)
        self.fc3_2 = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.to(device)
        self._initialize_weights()

    def forward(self, s, a):
        s = torch.cat([s, a], dim=1)

        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        values = self.fc3(x)

        x_2 = torch.relu(self.fc1_2(s))
        x_2 = torch.relu(self.fc2_2(x_2))
        values_2 = self.fc3_2(x_2)

        return values, values_2

    def _initialize_weights(self):
        """Kaiming Normalで初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # ReLUに適した初期化
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)  # バイアスはゼロで初期化
class TD3Agent:
    def __init__(self, state_dim, action_dim, device, tau=0.001, gamma=0.99):
        self.device = device
        self.action_dim = action_dim
        self.max_action = 2.0 # action range: [-2, 2]
        self.gamma=gamma
        self.tau=tau
        self.actor_network = ActorNetwork(state_dim=state_dim, action_dim=action_dim, device=device)
        self.target_actor_network = ActorNetwork(state_dim=state_dim, action_dim=action_dim, device=device)
        self.critic_network = CriticNetwork(state_dim=state_dim, action_dim=action_dim, device=device)
        self.target_critic_network = CriticNetwork(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor_optimizer = self.actor_network.optimizer
        self.critic_optimizer = self.critic_network.optimizer

        self._build_networks()

    def _build_networks(self):
        for target_param, param in zip(self.target_actor_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_(param.data)

    def policy(self, state, noise=None):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor_network(state_t).cpu().numpy().squeeze(0)

        # 2. Target Policy Smoothing
        if noise:
            action += np.random.normal(0, noise*self.max_action, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)

            # アクションを範囲内にクリップ
            action = np.clip(action, -self.actor_network.ACTION_RANGE, self.actor_network.ACTION_RANGE)

        return action

    def update(self, states, actions, rewards, next_states, dones, update_policy):
        with torch.no_grad():
            clipped_noise = torch.normal(mean=0.0, std=0.2, size=(self.action_dim,), device=self.device)
            clipped_noise = torch.clamp(clipped_noise, min=-0.5, max=0.5)  # Tensorとしてclampを適用

            next_actions = self.target_actor_network(next_states) + clipped_noise * self.max_action

            q1_next, q2_next = self.target_critic_network(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            target_values = rewards + self.gamma * (1 - dones) * min_q_next
    
        # Update Critic Network
        q1, q2 = self.critic_network(states, actions)
        critic_loss = nn.MSELoss()(q1, target_values) + nn.MSELoss()(q2, target_values)
    
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
        # Update Actor Network
        # 3. Delayed Policy Updates
        if not update_policy:
            return
        # ここではq1のみを用いて勾配を更新します
        # 方策関数（Actor）のパラメータを、Q関数 (Critic) の出力値が大きくなるように更新する
        actor_loss = -self.critic_network(states, self.actor_network(states))[0].mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
        # ターゲットネットワークをソフトアップデート
        self.update_target_network()


    def update_target_network(self):
        for target_param, param in zip(self.target_actor_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_critic_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, actor_model_path, critic_model_path):
        torch.save(self.actor_network.state_dict(), actor_model_path)
        torch.save(self.critic_network.state_dict(), critic_model_path)

    @classmethod
    def load(cls, actor_model_path, critic_model_path, state_dim, action_dim, device):
        agent = cls(state_dim=state_dim, action_dim=action_dim, device=device)
        agent.actor_network.load_state_dict(torch.load(actor_model_path, map_location=device))
        agent.critic_network.load_state_dict(torch.load(critic_model_path, map_location=device))
        agent._build_networks()
        return agent
