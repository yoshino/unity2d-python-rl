import numpy as np

from utils.udp_connection import UdpConnection

class CartPole:
    def __init__(self):
        self.actions = list(range(2)) #left, right
        self.feature_shape = 4        #cart_position, cart_velocity, pole_angle, pole_angular_velocity
        self.connection = UdpConnection()
        self.connection.connect()

    def receive(self):
        addr, data = self.connection.receive()
        state = self.transform(data['state'])
        reward = data['reward']
        done = (data['done'] == 1)
        world_time = data['worldTime']
        return addr, state, reward, done, world_time

    def transform(self, state):
        cart_position = state['cartPosition']
        cart_velocity = state['cartVelocity']
        pole_angle = state['poleAngle']
        pole_angular_velocity = state['poleAngularVelocity']

        state = [cart_position['x'], cart_velocity['x'], pole_angle, pole_angular_velocity]
        state = np.array(state, dtype=np.float32)
        return state

    def send(self, addr, action, world_time):
        self.connection.send(addr, {'action': int(action), 'worldTime': world_time})
