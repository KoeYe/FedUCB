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