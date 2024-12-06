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
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        self.reset_parameters()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.to(device)

    def reset_parameters(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        # 出力層は小さい範囲で初期化
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, s):
        s = s.to(self.device)
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        # 出力をtanhで[-1,1]にクリップし、その後ACTION_RANGEをかける
        actions = torch.tanh(self.fc3(x)) * self.ACTION_RANGE
        return actions

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)

        self.reset_parameters()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self.to(device)

    def reset_parameters(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, s, a):
        s, a = s.to(self.device), a.to(self.device)
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(torch.cat([x, a], dim=1)))
        values = self.fc3(x)
        return values

class OUNoise:
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class DDPGAgent:
    def __init__(self, state_dim, action_dim, device, tau=0.001, gamma=0.99):
        self.device = device
        self.action_dim = action_dim
        self.gamma=gamma
        self.tau=tau
        self.actor_network = ActorNetwork(state_dim=state_dim, action_dim=action_dim, device=device)
        self.target_actor_network = ActorNetwork(state_dim=state_dim, action_dim=action_dim, device=device)
        self.critic_network = CriticNetwork(state_dim=state_dim, action_dim=action_dim, device=device)
        self.target_critic_network = CriticNetwork(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor_optimizer = self.actor_network.optimizer
        self.critic_optimizer = self.critic_network.optimizer

        self._build_networks()

        self.noise = OUNoise(action_dimension=action_dim)

    def _build_networks(self):
        for target_param, param in zip(self.target_actor_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_(param.data)

    def policy(self, state, noise_scale=1.0):
        # 状態をテンソル化
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor_network(state_t).cpu().numpy().squeeze(0)
        # OUノイズを加える
        if noise_scale > 0:
            action += self.noise.sample() * noise_scale
        # アクションを範囲内にクリップ
        action = np.clip(action, -self.actor_network.ACTION_RANGE, self.actor_network.ACTION_RANGE)
        return action

    def update(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.target_actor_network(next_states)
            next_qvalues = self.target_critic_network(next_states, next_actions)
            target_values = rewards + self.gamma * (1 - dones) * next_qvalues

        # Update Critic Network
        qvalues = self.critic_network(states, actions)
        critic_loss = nn.MSELoss()(qvalues, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor Network
        actor_loss = -self.critic_network(states, self.actor_network(states)).mean()
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

    def reset_noise(self):
        self.noise.reset()
