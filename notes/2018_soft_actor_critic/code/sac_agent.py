import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# CRITIC CLASS
class ClippedCriticNet(nn.Module):

    def __init__(self, action_dim, state_dim, output_dim=1, hidden_size=256):

        super().__init__()

        input_dim = state_dim + action_dim

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_dim)

        self.linear4 = nn.Linear(input_dim, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, output_dim)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

# ACTOR CLASS: 方策ネットワーク
class SoftActorNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size, action_scale):

        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, output_dim)
        self.log_std_linear = nn.Linear(hidden_size, output_dim)

        self.action_scale = torch.tensor(action_scale)
        self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        # Reparameterization trick
        # 1. 状態ｓをガウス方策関数に与えてアクション分布の平均（μ）と標準偏差（σ）を得る
        # 2. 正規分布 N（μ、σ）からのサンプリングにより確率的にアクションaを決定する。
        # 3. 決定したアクションaと状態sをQ関数に与えてQ(s, a)を計算する
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias # シンプルなガウス分布と仮定している

        # Squashed Gaussian Policy
        # ガウス分布を-1から1の範囲に押しつぶされた（Squashed）ような分布に変換する
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def convert_network_grad_to_false(network):
    for param in network.parameters():
        param.requires_grad = False


class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound, device, tau=0.001, gamma=0.99):
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.gamma=gamma
        self.tau=tau
        self.alpha = 0.2

        # Actor Network
        # 確率的方策は任意の形式が使用可能ですが、論文で単ガウス方策が使用されているのでこれに倣います。
        # SAC論文の初期versionでは混合ガウス分布を使っていましたがmujuco環境では単ガウスでも混合ガウスでもあまりパフォーマンスに影響が無いようです。
        # REF: https://openreview.net/pdf?id=HJjvxl-Cb
        self.actor = SoftActorNet(
            input_dim=self.state_dim, output_dim=self.action_dim, hidden_size=256, action_scale=action_bound
        )
        self.actor.to(self.device)
        # Critic Network
        # TD3で提案されたClipped-Double-Qトリックを適用
        self.dualqnet = ClippedCriticNet(state_dim=self.state_dim, action_dim=self.action_dim, hidden_size=256).to(self.device)
        self.target_dualqnet = ClippedCriticNet(state_dim=self.state_dim, action_dim=self.action_dim, hidden_size=256).to(self.device)

        hard_update(self.target_dualqnet, self.dualqnet)
        convert_network_grad_to_false(self.target_dualqnet)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.dualqnet.parameters())

        # 温度パラメータαの最適化で利用する
        # ただしエントロピーの目標値Hはやはりハイパーパラメータであり、"-1×アクションの次元数" が推奨値として提案されているものの、
        # とくに理論的根拠があるわけではないのである程度ハイパラチューニングした方がよいと思われます。
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha])

    def policy(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not evaluate:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.cpu().detach().numpy().reshape(-1)

    def update(self, states, actions, rewards, next_states, masks):
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_states)
            next_q1_values_target, next_q2_values_target = self.target_dualqnet(next_states, next_action)
            next_q_values_target = torch.min(next_q1_values_target, next_q2_values_target) - self.alpha * next_log_pi
            next_q_values = rewards + masks * self.gamma * next_q_values_target

        q1_values, q2_values = self.dualqnet(states, actions)
        critic1_loss = F.mse_loss(q1_values, next_q_values)
        critic2_loss = F.mse_loss(q2_values, next_q_values)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action, log_pi, _ = self.actor.sample(states)

        q1_values, q2_values = self.dualqnet(states, action)
        q_values = torch.min(q1_values, q2_values)

        actor_loss = ((self.alpha * log_pi) - q_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        soft_update(self.target_dualqnet, self.dualqnet, self.tau)

        return critic_loss.item(), actor_loss.item()

    def save(self, model_path):
        torch.save({
            'log_alpha': self.log_alpha.detach().cpu().numpy(),
            'policy_state_dict': self.actor.state_dict(),
            'dualqnet_state_dict': self.dualqnet.state_dict(),
        }, model_path)


    @classmethod
    def load(cls, state_dim, action_dim, action_bound, device, model_path):
        agent = cls(state_dim, action_dim, action_bound, device)

        checkpoint = torch.load(model_path, map_location=device)
        agent.log_alpha.data = torch.tensor(checkpoint['log_alpha'], device=device)
        agent.actor.load_state_dict(checkpoint['policy_state_dict'])
        agent.dualqnet.load_state_dict(checkpoint['dualqnet_state_dict'])
        agent.target_dualqnet.load_state_dict(agent.dualqnet.state_dict())

        return agent

    def play(self, env, episode_count=10, render=True):
        self.actor.eval()

        for i in range(episode_count):
            s = env.reset()

            done = False
            episode_rewards = []

            while not done:
                s = torch.tensor(s, dtype=torch.float32).to(self.device)
                a, _ = self.actor(s)
                a = a.detach().cpu().numpy()

                n_state, reward, done = env.step(a)
                episode_rewards.append(reward)

                s = n_state

                if render:
                    env.render()

            # エピソード終了処理
            print(f'episode {i}: {sum(episode_rewards)}')
