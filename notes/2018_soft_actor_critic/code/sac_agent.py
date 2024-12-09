import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np


class GaussianPolicy(nn.Module):

    def __init__(self, action_space, action_bound, lr=3e-4):
        super(GaussianPolicy, self).__init__()
        self.action_space = action_space
        self.action_bound = action_bound

        self.dense1 = nn.Linear(3, 256)
        self.dense2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, self.action_space)
        self.logstd = nn.Linear(256, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        mu = torch.tanh(self.mu(x))
        logstd = self.logstd(x)
        return mu, logstd

    def sample_action(self, states):
        mu, logstd = self(states)
        std = torch.exp(logstd)

        # Reparameterization trick
        normal_noise = torch.randn_like(mu)
        actions = mu + std * normal_noise

        logprob = self._compute_logprob(mu, std, actions)

        # Squashed Gaussian Policy
        # 1. 正規分布を[-1, 1]の範囲にsquashする
        # 2. Actionの取りうる範囲self.action_boundをかける
        actions_squashed = torch.tanh(actions)
        logprob_squashed = logprob - torch.sum(
            torch.log(1 - torch.tanh(actions) ** 2 + 1e-6), dim=1, keepdim=True
        )

        actions_squashed *= self.action_bound

        return actions_squashed, logprob_squashed

    def _compute_logprob(self, means, stdevs, actions):
        logprob = -0.5 * np.log(2 * np.pi)
        logprob += -torch.log(stdevs)
        logprob += -0.5 * ((actions - means) / stdevs) ** 2
        logprob = torch.sum(logprob, dim=1, keepdim=True)
        return logprob


class DualQNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, lr=3e-4):
        super(DualQNetwork, self).__init__()
        input_dim = state_dim + action_dim 
        self.dense_11 = nn.Linear(input_dim, 256)
        self.dense_12 = nn.Linear(256, 256)
        self.q1 = nn.Linear(256, 1)

        self.dense_21 = nn.Linear(input_dim, 256)
        self.dense_22 = nn.Linear(256, 256)
        self.q2 = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, states, actions):
        inputs = torch.cat([states, actions], dim=1)

        x1 = torch.relu(self.dense_11(inputs))
        x1 = torch.relu(self.dense_12(x1))
        q1 = self.q1(x1)

        x2 = torch.relu(self.dense_21(inputs))
        x2 = torch.relu(self.dense_22(x2))
        q2 = self.q2(x2)

        return q1, q2


class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound, device, tau=0.001, gamma=0.99):
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.gamma=gamma
        self.tau=tau

        # Actor Network
        # 確率的方策は任意の形式が使用可能ですが、論文で単ガウス方策が使用されているのでこれに倣います。
        # SAC論文の初期versionでは混合ガウス分布を使っていましたがmujuco環境では単ガウスでも混合ガウスでもあまりパフォーマンスに影響が無いようです。
        # REF: https://openreview.net/pdf?id=HJjvxl-Cb
        self.policy = GaussianPolicy(
            action_space=self.action_dim, action_bound=self.action_bound
        ).to(self.device)
        # Critic Network
        # TD3で提案されたClipped-Double-Qトリックを適用
        self.duqlqnet = DualQNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.target_dualqnet = DualQNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)

        # 温度パラメータαの最適化で利用する
        # ただしエントロピーの目標値Hはやはりハイパーパラメータであり、"-1×アクションの次元数" が推奨値として提案されているものの、
        # とくに理論的根拠があるわけではないのである程度ハイパラチューニングした方がよいと思われます。
        self.target_entropy = -0.5 * self.action_dim
        self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True, device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def policy(self, states):
        states = states.to(self.device)
        return self.policy.sample_action(states)

    def update(self, states, actions, rewards, next_states, dones):
        alpha = torch.exp(self.log_alpha)
    
        # --- 1. Update Q-function ---
        # 次状態からpolicyによるサンプル
        with torch.no_grad():
            next_actions, next_logprobs = self.policy(next_states) # next_logprobs: logπ(a|s) : エントロピーボーナス
            target_q1, target_q2 = self.target_dualqnet(next_states, next_actions)
            target = rewards + (1 - dones) * self.gamma * (torch.min(target_q1, target_q2) - alpha * next_logprobs)
    
        q1, q2 = self.duqlqnet(states, actions)
        q_loss_1 = F.mse_loss(q1, target)
        q_loss_2 = F.mse_loss(q2, target)
        q_loss = 0.5 * (q_loss_1 + q_loss_2)
    
        self.duqlqnet.optimizer.zero_grad()
        q_loss.backward()
        self.duqlqnet.optimizer.step()
    
        # --- 2. Update policy ---
        # 方策関数はsoft-Q関数のSoftmax方策に似せていく、つまりKL距離を最小化することにより最適方策が得られる
        # soft-Q関数のSoftmax方策 と 方策関数のKL距離の最小化は、最終的に　「policy_loss = (alpha * logprobs - q_min).mean()」となる
        actions_sampled, logprobs = self.policy(states)
        q1_new, q2_new = self.duqlqnet(states, actions_sampled)
        q_min = torch.min(q1_new, q2_new)
        policy_loss = (alpha * logprobs - q_min).mean()
    
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()
    
        # --- 3. Adjust alpha ---
        # 最初のSAC論文ではこの係数αはハイパーパラメータとして調整されるべき値とされましたが、
        # SAC論文② では係数αの自動調整手法が提案されています。
        # 再度サンプリング（元コードに合わせて再サンプル）
        with torch.no_grad():
            _, logprobs_for_alpha = self.policy(states)
        entropy_diff = -logprobs_for_alpha - self.target_entropy
    
        alpha_loss = (torch.exp(self.log_alpha) * entropy_diff).mean()
    
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
    
        # --- Soft target update ---
        with torch.no_grad():
            for target_param, param in zip(self.target_dualqnet.parameters(), self.duqlqnet.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, model_path):
        torch.save({
            'log_alpha': self.log_alpha.detach().cpu().numpy(),
            'policy_state_dict': self.policy.state_dict(),
            'duqlqnet_state_dict': self.duqlqnet.state_dict(),
        }, model_path)


    @classmethod
    def load(cls, state_dim, action_dim, action_bound, device, model_path):
        agent = cls(state_dim, action_dim, action_bound, device)

        checkpoint = torch.load(model_path, map_location=device)
        agent.log_alpha.data = torch.tensor(checkpoint['log_alpha'], device=device)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.duqlqnet.load_state_dict(checkpoint['duqlqnet_state_dict'])
        agent.target_duqlqnet.load_state_dict(self.duqlqnet.state_dict())

        return agent
