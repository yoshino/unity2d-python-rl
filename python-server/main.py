import socket

def start_udp_server(host='127.0.0.1', port=12345):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"Listening on {host}:{port}")

    while True:
        data, addr = sock.recvfrom(1024)
        if not data:
            break
        
        message = data.decode('utf-8')
        print(f"Received message: {message}")
        
        # 位置情報を解析して1を加える
        positions = message.split(',')
        if len(positions) == 3:
            try:
                x, y, z = float(positions[0]), float(positions[1]), float(positions[2])
                x += 1
                y += 1
                z += 1
                response = f"{x},{y},{z}"
            except ValueError:
                response = "Invalid data"
        else:
            response = "Invalid format"

        sock.sendto(response.encode('utf-8'), addr)

if __name__ == "__main__":
    start_udp_server()
