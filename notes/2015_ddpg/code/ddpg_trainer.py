import re
import random
import numpy as np
from collections import deque, namedtuple
import torch

from .ddpg_agent import DDPGAgent
from .replay_buffer import ReplayMemory

class DDPGTrainer():
    def __init__(
        self, model_path, device, 
        buffer_size=100000, 
        batch_size=256, tau=0.005, gamma=0.99,
        warmup_steps=1000
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.episode_count = 100
        self.memory = ReplayMemory(buffer_size)
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
        agent = DDPGAgent(state_dim=3, action_dim=env.action_space, device=self.device, tau=self.tau, gamma=self.gamma)
        agent.is_training = True
        
        # warmup前はランダムアクション、それ以降はpolicyにしたがって行動
        self.train_loop(env=env, agent=agent)
        return agent

    def train_loop(self, env, agent):
        for i in range(self.episode_count):
            s = env.reset()
            agent.reset_noise()  # エージェント側でノイズリセット
            done = False
            episode_reward = 0.0

            while not done:
                if self.total_steps < self.warmup_steps:
                    # warmup期間はランダム行動
                    a = np.random.uniform(-2, 2, size=env.action_space)
                else:
                    # warmup終了後はpolicyに従った行動（内部でノイズ適用）
                    a = agent.policy(s)

                n_state, reward, done = env.step(a)

                if len(self.memory) > self.batch_size:
                    states, actions, rewards, next_states, masks = self.memory.sample(batch_size=self.batch_size)
                    states = torch.FloatTensor(states).to(self.device)
                    next_states = torch.FloatTensor(next_states).to(self.device)
                    actions = torch.FloatTensor(actions).to(self.device)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
                    masks = torch.FloatTensor(masks).unsqueeze(1).to(self.device)
                    agent.update(states, actions, rewards, next_states, masks)

                episode_reward += reward
                self.total_steps += 1
                self.episode_rewards.append(reward)
                self.memory.push(state=s, action=a, reward=reward, next_state=n_state, mask=float(not done))

                s = n_state

            # エピソード終了処理
            self.episode_end(episode=i, agent=agent, episode_reward=episode_reward)

        actor_model_path = self.model_path.replace('.pth', '_{}.pth'.format('final_actor'))
        critic_model_path = self.model_path.replace('.pth', '_{}.pth'.format('final_critic'))
        agent.save(actor_model_path=actor_model_path, critic_model_path=critic_model_path)
        print('final model saved: ', actor_model_path + ' and ' + critic_model_path)

    def episode_end(self, episode, agent, episode_reward):
        sum_reward = sum(self.episode_rewards)
        print('episode: {}, sum_reward: {}'.format(episode, sum_reward))

        if sum_reward > self.max_episode_reward:
            actor_model_path = self.model_path.replace('.pth', '_{}.pth'.format('best_actor'))
            critic_model_path = self.model_path.replace('.pth', '_{}.pth'.format('best_critic'))
            agent.save(actor_model_path=actor_model_path, critic_model_path=critic_model_path)
            print('best model saved: ', actor_model_path + ' and ' + critic_model_path)
            self.max_episode_reward = sum_reward

        # 次エピソードに備えてリワードリストをクリア
        self.episode_rewards = []
        # 環境やエージェント内部状態のリセット
        agent.reset_noise()
