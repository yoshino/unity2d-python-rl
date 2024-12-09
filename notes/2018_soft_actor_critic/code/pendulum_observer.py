
import torch
import numpy as np
import gymnasium as gym

class PendulumObserver():
    def __init__(self, play=False):
        if play:
            self._env = gym.make("Pendulum-v1", render_mode="human")
        else:
            self._env = gym.make("Pendulum-v1")

    @property
    def action_space(self):
        return 1

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        s, _ = self._env.reset()
        return s

    def render(self):
        self._env.render()

    def step(self, action):
        # エピソードの終了条件はなく、総時刻(行動回数)が200を超えると、
        # 打ち切りフラグtruncatedがTrue(エピソードが打ち切り)になります。
        state, reward, _, done, _ = self._env.step(action)

        return state, reward, done
