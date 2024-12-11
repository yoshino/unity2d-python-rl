import re
import random
import torch
import numpy as np
from collections import deque, namedtuple

from .sac_agent import SACAgent
from .replay_buffer import ReplayBuffer
from .replay_buffer import ReplayMemory

class SACTrainer():
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
        self.warmup_steps = warmup_steps  # warmup期間の導入
        self.env = None
        self.eval_interval = 10

    def train(self, env):
        self.env = env
        agent = SACAgent(state_dim=3, action_dim=env._env.action_space.shape[0], action_bound=2.0, device=self.device, tau=self.tau, gamma=self.gamma)
        
        # warmup前はランダムアクション、それ以降はpolicyにしたがって行動
        self.train_loop(env=env, agent=agent)
        return agent

    def train_loop(self, env, agent):
        for i in range(self.episode_count):
            s = env.reset()

            done = False
            episode_reward = 0.0

            while not done:
                # s = torch.tensor(s, dtype=torch.float32).to(self.device)
                # s = s.unsqueeze(0)  # バッチ次元を追加( shape: [1, state_dim] )

                if self.total_steps > self.warmup_steps:
                    a = agent.policy(s)
                else:
                    a = env._env.action_space.sample()

                if len(self.memory) > self.batch_size:
                    states, actions, rewards, next_states, masks = self.memory.sample(batch_size=self.batch_size)
                    states = torch.FloatTensor(states).to(self.device)
                    next_states = torch.FloatTensor(next_states).to(self.device)
                    actions = torch.FloatTensor(actions).to(self.device)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
                    masks = torch.FloatTensor(masks).unsqueeze(1).to(self.device)
                    agent.update(states, actions, rewards, next_states, masks)

                n_state, reward, done = env.step(a)
                self.total_steps += 1
                episode_reward += reward
                self.memory.push(state=s, action=a, reward=reward, next_state=n_state, mask=float(not done))
                self.episode_rewards.append(reward)

                s = n_state

            # エピソード終了処理
            self.episode_end(episode=i, agent=agent, episode_reward=episode_reward, env=env)

        model_path = self.model_path.replace('.pth', '_{}.pth'.format('final'))
        agent.save(model_path)
        print('final model saved: ', model_path)

    def episode_end(self, episode, agent, episode_reward, env):
        sum_reward = sum(self.episode_rewards)
        print('episode: {}, sum_reward: {}'.format(episode, sum_reward))

        if sum_reward > self.max_episode_reward:
            model_path = self.model_path.replace('.pth', '_{}.pth'.format('best'))
            agent.save(model_path)
            print('best model saved: ', model_path)
            self.max_episode_reward = sum_reward


        # 次エピソードに備えてリワードリストをクリア
        self.episode_rewards = []

        if episode % self.eval_interval == 0:
            avg_reward = 0.
            for _  in range(self.eval_interval):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    # state = torch.tensor(state, dtype=torch.float32).to(self.device)
                    # state  = state.unsqueeze(0)  # バッチ次元を追加( shape: [1, state_dim] )
                    with torch.no_grad():
                        action = agent.policy(state, evaluate=True)
                    n_state, reward, done = env.step(action)

                    episode_reward += reward
                    state = n_state
                avg_reward += episode_reward
            avg_reward /= self.eval_interval

            print("Episode: {}, Eval Avg. Reward: {:.0f}".format(episode, avg_reward))
