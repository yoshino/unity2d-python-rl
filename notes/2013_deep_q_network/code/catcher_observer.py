from PIL import Image # opencvは処理が重いので使わない方が良いらしい
import torch
import numpy as np

class CatcherObserver():
    def __init__(self, env, width, height, n_frame, image_size=84):
        self._env = env
        self.width = width
        self.height = height
        self.n_frame = n_frame
        self.image_size = image_size
        self.state_frames = None
        self.lives = 5

    def transform(self, state, done):
        # Pillowを使ってグレースケール化とリサイズを行う
        state_image = Image.fromarray(state[32:, 8:152])  # 必要な範囲をトリミング
        state_image = state_image.convert('L')  # グレースケール変換
        state_image = state_image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)  # リサイズ

        # 画像をnumpy配列として取得し、正規化
        observation_frame = np.asarray(state_image, dtype=np.float32) / 255.0

        # Tensorに変換
        observation_frame = torch.tensor(observation_frame, dtype=torch.float32)

        # 初期化時に最初のフレームをコピーしてstate_framesを作成
        if self.state_frames is None:
            self.state_frames = observation_frame.repeat((self.n_frame, 1, 1))

        # フレームをロールし、最新のフレームを追加
        self.state_frames = torch.roll(self.state_frames, shifts=-1, dims=0)
        self.state_frames[-1, :, :] = observation_frame

        # バッチ次元とチャネル次元を追加して返す
        return self.state_frames

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        s, info = self._env.reset()
        self.lives = 5
        return self.transform(s, done=False)

    def render(self):
        self._env.render()

    def step(self, action):
        n_state, reward, done, _, info = self._env.step(action)

        # Breakoutにおける１エピソードとは５つの残機をすべて失いGam eoverとなるまでです。
        # 当然、DQN論文に表記されているスコアもGame overになるまでに獲得したトータルスコアです。
        # しかし、それでは残機を失うことが悪いことだとエージェントが学習しにくくなってしまうので、
        # DQNでは残機が減ったら遷移情報（transition）ではエピソードが終了した扱いにするというトリックが使用されています。
        # 最近の手法（R2D2）では、このトリックを使わないことによりatari環境の一部のゲーム(seaquest)で
        # 大幅に性能改善が見られることが報告されていますが、Breakoutについてはこのトリックを使用した方が学習の進みが良いです。
        if info["lives"] != self.lives:
            self.lives = info["lives"]
            done = True

        return self.transform(n_state, done), reward, done, info
