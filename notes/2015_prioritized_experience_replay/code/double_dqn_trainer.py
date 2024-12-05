import re
import random
from collections import deque, namedtuple

from .double_dqn_agent import DoubleDqnAgent
from .prioritized_replay_buffer import PrioritizedReplayBuffer


class DoubleDqnTrainer():
    def __init__(self, model_path, device, buffer_size=1000000, batch_size=32, gamma=0.99):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_scheduler = (
           lambda steps: max(1.0 - 0.9 * steps / buffer_size, 0.1))
        self.episode_count = 50000
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size, device=device)
        self.training = False
        self.model_path = model_path
        self.model_update_freq = 4
        self.teacher_model_update_freq = 10000
        self.model_save_freq = 1000
        self.device = device
        self.max_episode_reward = -10
        self.episode_loss = []
        self.episode_rewards = []
        self.total_steps = 0

    def train(self, env):
        agent = DoubleDqnAgent(epsilon=self.epsilon_scheduler(self.total_steps), actions=list(range(env.action_space.n)), device=self.device)  
        observe_interval = 0
        self.train_loop(env=env, agent=agent)
        return agent

    def train_loop(self, env, agent):
        for i in range(self.episode_count):
            s = env.reset()
            done = False

            while not done:
                a = agent.policy(s)

                # 経験に基づいてパラメータを更新する
                n_state, reward, done, info = env.step(a)
                self.total_steps += 1

                transition = (s, a, reward, n_state, done)
                self.replay_buffer.push(transition)
                self.episode_rewards.append(reward)

                if len(self.replay_buffer) > 50000:
                    # モデルの初期化
                    if not self.training:
                        agent.initialize()
                        self.episode_loss = []
                        self.episode_rewards = []
                        self.training = True

                    #: 4ステップごとにQネットワークを更新
                    if self.total_steps % self.model_update_freq == 0:
                        indices, weights, (states, actions, rewards, next_states, dones) = self.replay_buffer.get_minibatch(
                            self.batch_size, self.total_steps
                        )
                        loss, td_error_abs  = agent.update(weights, states, actions, rewards, next_states, dones)
                        self.replay_buffer.update_priorities(indices=indices, td_errors=td_error_abs)

                        self.episode_loss.append(loss)
                        agent.epsilon = self.epsilon_scheduler(self.total_steps)


                    #: 10000ステップごとにtarget-QネットワークをQネットワークと同期
                    if self.total_steps % self.teacher_model_update_freq == 0:
                        agent.update_teacher()

                s = n_state
            else:
                self.episode_end(episode=i, agent=agent)

    def episode_end(self, episode, agent):
        if not self.training or len(self.episode_loss) == 0:
            return

        avg_loss = sum(self.episode_loss) / len(self.episode_loss)
        avg_loss = round(avg_loss, 2)

        sum_reward = sum(self.episode_rewards)

        if sum_reward > self.max_episode_reward:
            model_path = self.model_path.replace('.pth', '_{}.pth'.format('best'))
            agent.save(model_path)
            print('episode: {}, step_count: {}, sum_reward: {}, avg_loss: {}'.format(episode, len(self.episode_rewards), sum_reward, avg_loss))
            print('best model saved: ', model_path)
            self.max_episode_reward = sum_reward

        if self.training and self.total_steps % self.model_save_freq == 0:
            model_path = self.model_path.replace('.pth', '_{}.pth'.format(episode))
            agent.save(model_path)
            print('episode: {}, step_count: {}, sum_reward: {}, avg_loss: {}'.format(episode, len(self.episode_rewards), sum_reward, avg_loss))
            print('model saved: ', model_path)

        self.episode_loss = []
        self.episode_rewards = []
