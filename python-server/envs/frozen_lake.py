from utils.udp_connection import UdpConnection

class FrozenLake:
    def __init__(self, grid_size=4):
        self.actions = list(range(4))
        self.grid_size = grid_size
        self.connection = UdpConnection()
        self.connection.connect()

    def receive(self):
        addr, data = self.connection.receive()
        state = data['state']
        reward = data['reward']
        done = (data['done'] == 1)
        world_time = data['worldTime']
        return addr, state, reward, done, world_time

    def send(self, addr, action, world_time):
        self.connection.send(addr, {'action': int(action), 'worldTime': world_time})
