import socket
import threading
import struct

from .model import Model
from .config import SERVER_HOST, SERVER_PORT, logger
from .protocol import Protocol

class Client:
    def __init__(self):
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.connect((SERVER_HOST, SERVER_PORT))
        self._client.setblocking(False)
        # 这里初始化了model，但是参数要等到server端来初始化
        self._model = Model()
        self._protocol = Protocol()

    def send(self, data):
        self._client.send(data.encode())

    def recv_all(self, length):
        data = b""
        while len(data) < length:
            more = self._client.recv(length - len(data))
            if not more:
                return b""
            data += more
        return data

    def recv(self):
        try:
            length = self.recv_all(4)
            logger.debug(length)
            if not length:
                return b""
            length = struct.unpack('i', length)[0]
            data = self.recv_all(length)
            logger.debug(data)
            if not data:
                return b""
            type, message = self._protocol.decode(data)
            if type == 'text':
                logger.info("Received data: %s", message)
            elif type == 'params':
                logger.info("aligning parameters")
                self.align_param(message)
        except BlockingIOError:
            pass

    def align_param(self, data):
        logger.debug("parameters: \n" + str(self._model.state_dict()))
        data_dict = {}
        for i in self._model.state_dict().keys():
            data_dict[i] = data.pop(0)
        self._model.load_state_dict(data_dict)
        logger.debug("aligned parameters: \n" + str(self._model.state_dict()))

    def run(self):
        logger.info("Client started")
        while True:
            try:
                self.recv()
            except KeyboardInterrupt:
                break
            # TODO: training and send model gradient to server, then update model parameters

    def close(self):
        self._client.close()