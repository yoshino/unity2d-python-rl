import numpy as np
import pickle
import zlib
from dataclasses import dataclass
import torch

@dataclass
class Experience:
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # リーフノード（経験）の最大数
        self.tree = np.zeros(2 * capacity - 1)  # ツリーの全ノード数
        self.data = [None] * capacity  # 実際の経験データ
        self.write = 0  # 次に書き込むインデックス

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2  # 親ノードのインデックス
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1  # 左の子ノード
        right = left + 1  # 右の子ノード

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]  # 全優先度の合計

    def add(self, p, data):
        idx = self.write + self.capacity - 1  # リーフノードのインデックス
        self.data[self.write] = data  # データの格納
        self.update(idx, p)  # 優先度の更新

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0  # バッファが満杯の場合は上書き

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p  # 優先度の更新

        self._propagate(idx, change)  # 変更を伝播

    def get(self, s):
        idx = self._retrieve(0, s)  # リーフノードのインデックス取得
        dataIdx = idx - self.capacity + 1  # データのインデックス
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:

    def __init__(self, capacity, device='cpu', reward_clip=True, alpha=0.6, beta=0.4,
                 total_steps=2500000, compress=True):

        self.capacity = capacity
        self.device = device
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_scheduler = (lambda steps: beta + (1 - beta) * steps / total_steps)
        self.epsilon = 0.01
        self.max_priority = 1.0
        self.reward_clip = reward_clip
        self.compress = compress

    def __len__(self):
        return self.tree.write if self.tree.write != 0 else self.capacity

    def push(self, transition):
        """
        Args:
            transition : tuple(state, action, reward, next_state, done)
        """
        exp = Experience(*transition)
        exp.reward = np.clip(exp.reward, -1, 1) if self.reward_clip else exp.reward

        if self.compress:
            exp = zlib.compress(pickle.dumps(exp))

        priority = self.max_priority  # 新しい経験には最大の優先度を割り当てる

        self.tree.add(priority, exp)

    def get_minibatch(self, batch_size, steps):
        indices = []
        experiences = []
        priorities = []

        N = self.capacity if self.tree.write >= self.capacity else self.tree.write

        beta = self.beta_scheduler(steps)
        # 優先度の合計をバッチサイズで分割
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            if self.compress:
                exp = pickle.loads(zlib.decompress(data))
            else:
                exp = data
            experiences.append(exp)

        # サンプリング確率の計算
        probabilities = np.array(priorities) / self.tree.total()
        # 補正重みの計算(モデルの損失の計算時に利用する)
        weights = (probabilities * N) ** (-beta)
        weights /= weights.max()
        weights = weights.reshape(-1, 1).astype(np.float32)
        weights = torch.from_numpy(weights).squeeze(1).to(self.device)

        states = torch.stack([exp.state for exp in experiences]).to(self.device)

        actions = [exp.action for exp in experiences]
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)

        rewards = [exp.reward for exp in experiences]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)

        dones = [exp.done for exp in experiences]
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return indices, weights, (states, actions, rewards, next_states, dones)

    def update_priorities(self, indices, td_errors):
        """
        Args:
            indices : 1D-array
            td_errors : 1D-array
        """
        # p(i) = (|δ|+ε)^α
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        for idx, priority in zip(indices, priorities):
            if priority > self.max_priority:
                self.max_priority = priority
            self.tree.update(idx, priority)
