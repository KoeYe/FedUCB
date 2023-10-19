import socket

from .configure import SERVER_HOST
from .configure import SERVER_PORT

class Client:
    def __init__(self):
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.connect((SERVER_HOST, SERVER_PORT))
        self._client.setblocking(False)

    def send(self, data):
        self._client.send(data.encode())

    def recv(self):
        try:
            data = self._client.recv(1024)
            if data:
                print("Received from server:", data.decode())
        except BlockingIOError:
            pass
        except Exception as e:
            print(e)

    def run(self):
        while True:
            try:
                data = input("Enter data to send: ")
                self.send(data)
                self.recv()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)

    def close(self):
        self._client.close()