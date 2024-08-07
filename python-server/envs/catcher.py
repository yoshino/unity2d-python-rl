import torch
import cv2
import base64
import numpy as np

from utils.udp_connection import UdpConnection

class Catcher:
    def __init__(self, n_frame=4, image_size=84):
        self.actions = list(range(3))
        self.connection = UdpConnection()
        self.connection.connect()
        self.image_size = image_size
        self.state_frames = torch.zeros((n_frame, image_size, image_size), dtype=torch.float32)

    def receive(self):
        addr, data = self.connection.receive()
        state = self.transform(data['screenShot'])
        reward = data['reward']
        done = (data['done'] == 1)
        world_time = data['worldTime']
        return addr, state, reward, done, world_time

    def send(self, addr, action, world_time):
        self.connection.send(addr, {'action': int(action), 'worldTime': world_time})

    def transform(self, state):
        image_bytes = base64.b64decode(state)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decoding failed.")

        observation_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
        observation_frame = cv2.resize(src=observation_frame[32:,8:152], dsize=(self.image_size, self.image_size))
        observation_frame = torch.tensor(observation_frame, dtype=torch.float32)

        # フレームをロールし、最新のフレームを追加
        self.state_frames = torch.roll(self.state_frames, shifts=-1, dims=0)
        self.state_frames[-1, :, :] = observation_frame

        # バッチ次元とチャネル次元を追加して返す
        return self.state_frames
