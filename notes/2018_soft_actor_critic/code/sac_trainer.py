import re
import random
import torch
import numpy as np
from collections import deque, namedtuple

from .sac_agent import SACAgent
from .replay_buffer import ReplayBuffer

class SACTrainer():
    def __init__(
        self, model_path, device, 
        buffer_size=100000, 
        batch_size=128, tau=0.005, gamma=0.99,
        warmup_steps=512
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.episode_count = 2000
        self.replay_buffer = ReplayBuffer(max_len=buffer_size, device=device)
        self.model_path = model_path
        self.device = device
        self.max_episode_reward = -10
        self.episode_rewards = []
        self.total_steps = 0
        self.noise_stdev = 0.2
        self.warmup_steps = warmup_steps  # warmup期間の導入
        self.env = None

    def train(self, env):
        self.env = env
        agent = SACAgent(state_dim=3, action_dim=env.action_space, action_bound=2.0, device=self.device, tau=self.tau, gamma=self.gamma)
        
        # warmup前はランダムアクション、それ以降はpolicyにしたがって行動
        self.train_loop(env=env, agent=agent)
        return agent

    def train_loop(self, env, agent):
        for i in range(self.episode_count):
            s = env.reset()

            done = False
            episode_reward = 0.0
            episode_step_count = 0

            while not done:
                s = torch.tensor(s, dtype=torch.float32).to(self.device)
                a, _ = agent.policy(s)
                a = a.detach().cpu().numpy() 
                n_state, reward, done = env.step(a)
                self.total_steps += 1
                episode_step_count += 1
                episode_reward += reward
                transition = (s, a, reward, n_state, done)
                self.replay_buffer.push(transition)
                self.episode_rewards.append(reward)

                if self.total_steps > self.warmup_steps:
                    states, actions, rewards, next_states, dones = self.replay_buffer.get_minibatch(self.batch_size)
                    agent.update(states, actions, rewards, next_states, dones)

                s = n_state

            # エピソード終了処理
            self.episode_end(episode=i, agent=agent, episode_reward=episode_reward, episode_step_count=episode_step_count)

        model_path = self.model_path.replace('.pth', '_{}.pth'.format('final'))
        agent.save(model_path)
        print('final model saved: ', model_path)

    def episode_end(self, episode, agent, episode_reward, episode_step_count):
        sum_reward = sum(self.episode_rewards)
        print('episode: {}, step_count: {}, sum_reward: {}'.format(episode, episode_step_count, sum_reward))

        if sum_reward > self.max_episode_reward:
            model_path = self.model_path.replace('.pth', '_{}.pth'.format('best'))
            agent.save(model_path)
            print('best model saved: ', model_path)
            self.max_episode_reward = sum_reward

        # 次エピソードに備えてリワードリストをクリア
        self.episode_rewards = []
