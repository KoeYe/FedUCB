import socket
import threading
import time

from .model import Model
from .config import SERVER_HOST, SERVER_PORT, logger

class Server:
    def __init__(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setblocking(False)
        self._punctuating = False
        self._client_num = 0
        self._clients = []
        self._model = Model()
        # self.print_model()

    def print_model(self):
        for name, param in self._model.named_parameters():
            logger.debug(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
            logger.debug("Values: ")
            logger.debug(param.data)

    def add_client(self):
        try:
            client, address = self._server.accept()
            self._clients.append(client)
            self._client_num += 1
            logger.info("New client connected: %s", address)
            logger.info("Current clients number: %d", self._client_num)
            # send model parameters to client
            self.send_param()
        except BlockingIOError:
            pass
        except Exception as e:
            print(e)

    def send_param(self):
        param = self._model.parameters()
        self.send(param)

    def remove_client(self, client):
        logger.info("Client disconnected")
        self._clients.remove(client)
        self._client_num -= 1
        logger.info("Current clients number: %d", self._client_num)

    def close(self):
        logger.debug("Closing server...")
        for client in self._clients:
            client.close()
        self._server.close()
        self._punctuating = False
        logger.debug("Server closed")

    def send(self, data):
        for client in self._clients:
            try:
                client.send(data)
            except Exception as e:
                self.remove_client(client)
                print(e)

    def handle_client(self, client):
        try:
            data = client.recv(1024)
            if data:
                logger.info("Received from client: %s", data.decode())
        except BlockingIOError:
            pass
        except Exception as e:
            self.remove_client(client)
            print(e)

    # keep clients alive
    def punctuate(self):
        # send data to clients every 1 sec
        while True:
            try:
                self.send(time.ctime().encode())
                time.sleep(1)
            except KeyboardInterrupt:
                break

    # boost the server
    def run(self, server_host=SERVER_HOST, server_port=SERVER_PORT):
        # bind and listen
        self._server.bind((server_host, server_port))
        self._server.listen(5)
        logger.info("Server is listening on %s:%s", server_host, server_port)

        # main loop
        while True:
            # get new client
            try:
                self.add_client()
            except KeyboardInterrupt:
                break

            # punctuate clients
            if not self._punctuating:
                self._punctuating = True
                threading.Thread(target=self.punctuate).start()

            # receive data from clients
            # TODO: add aggregation operation, then update model parameters, then send parameter to clients
            for client in self._clients:
                self.handle_client(client)

        # close server
        self.close()