# REF: https://github.com/horoiwa/deep_reinforcement_learning_gallery/blob/master/DQN/BreakoutDet-v4/buffer.py

from dataclasses import dataclass
import numpy as np
import pickle
import zlib
import torch

@dataclass
class Experience:

    state: np.ndarray

    action: float

    reward: float

    next_state: np.ndarray

    done: bool


class ReplayBuffer:

    def __init__(self, max_len, device, compress=True):
        self.max_len = max_len
        self.device = device
        self.buffer = []
        self.compress = compress
        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        """
            transition : tuple(state, action, reward, next_state, done)
        """

        # NumPy配列をTensorに変換して保存
        state, action, reward, next_state, done = transition
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        exp = Experience(state, action, reward, next_state, done)

        if self.compress:
            exp = zlib.compress(pickle.dumps(exp))

        if self.count == self.max_len:
            self.count = 0

        try:
            self.buffer[self.count] = exp
        except IndexError:
            self.buffer.append(exp)

        self.count += 1

    def get_minibatch(self, batch_size):

        N = len(self.buffer)

        indices = np.random.choice(
            np.arange(N), replace=False, size=batch_size)

        if self.compress:
            selected_experiences = [
                pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]
        else:
            selected_experiences = [self.buffer[idx] for idx in indices]

        # 既にTensorとして保存しているのでそのままstack
        states = torch.stack([exp.state for exp in selected_experiences]).to(self.device)
        actions = torch.tensor([exp.action for exp in selected_experiences], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([exp.reward for exp in selected_experiences], dtype=torch.float32, device=self.device)
        next_states = torch.stack([exp.next_state for exp in selected_experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in selected_experiences], dtype=torch.float32, device=self.device)

        return (states, actions, rewards, next_states, dones)
