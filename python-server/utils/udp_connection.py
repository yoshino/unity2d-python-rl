import socket
import json

class UdpConnection:
    def __init__(self):
        self.server_ip = '0.0.0.0'
        self.server_port = 9000
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def connect(self):
        self.sock.bind((self.server_ip, self.server_port))

    def receive(self):
        data, addr = self.sock.recvfrom(65507)  # 受信バッファサイズを増加
        return addr, json.loads(data.decode('utf-8'))

    def send(self, addr, data):
        response = json.dumps(data)
        self.sock.sendto(response.encode('utf-8'), addr)
