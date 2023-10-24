import socket

from .model import Model
from .config import SERVER_HOST, SERVER_PORT, logger

class Client:
    def __init__(self):
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.connect((SERVER_HOST, SERVER_PORT))
        self._client.setblocking(False)
        # 这里初始化了model，但是参数要等到server端来初始化
        self._model = Model()

    def send(self, data):
        self._client.send(data.encode())

    def recv(self):
        try:
            data = self._client.recv(1024)
            if data:
                logger.info("Received data: %s", data)
        except BlockingIOError:
            pass

    def run(self):
        while True:
            try:
                self.recv()
            except KeyboardInterrupt:
                break
            # TODO: training and send model gradient to server, then update model parameters



    def close(self):
        self._client.close()