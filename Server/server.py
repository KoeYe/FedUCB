import socket
import threading
import time
import struct
from torch import optim

from .model import Model
from .config import SERVER_HOST, SERVER_PORT, LEARNING_RATE, MOMENTUM, BANDWIDTH, TEST_DATA_DIR
from .protocol import Protocol
from .util import read_data, logger

class Server:
    def __init__(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setblocking(False)
        self._punctuating = False
        self._client_num = 0
        self._clients = []
        self._model = Model()
        self._optimizer = optim.SGD(self._model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        # self.print_model()
        self._protocol = Protocol()
        self._test_data = read_data(TEST_DATA_DIR)

    def print_model(self):
        for name, param in self._model.named_parameters():
            logger.debug(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
            logger.debug("Values: ")
            logger.debug(param.data)

    def add_client(self):
        # print("Waiting for new client...")
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
        logger.info("Sending parameters")
        # logger.debug("parameters: \n" + str(self._model.state_dict()))
        param_values = [param.data for param in self._model.parameters()]
        message = self._protocol.encode_parameters(param_values, 'params')
        self.send(message)

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
                print(e)
                self.remove_client(client)

    def handle_client(self, client):
        length = None
        while not length:
            try:
                length = client.recv(4)
            except BlockingIOError:
                continue
            except Exception as e:
                print(e)
                self.remove_client(client)
                return
        print("length: ", str(struct.unpack('i', length)[0]))
        data = client.recv(struct.unpack('i', length)[0])
        type, message = self._protocol.decode(data)
        print(type)
        if type == 'grad':
            logger.info("Received gradients")
            self._model.update_model_with_gradients(message, self._optimizer)
            self.send_param()
            time.sleep(0.1)
            # test model
            acc, loss = self._model.test(self._test_data)
            logger.info("Test accuracy: %f", acc)
            logger.info("Test loss: %f", loss)
        elif type == 'text':
            logger.info("Received data: %s", message)
        elif type == 'done':
            logger.info("Client finished")
            self.remove_client(client)
        else:
            logger.info("Unknown message type: " + type)

    # keep clients alive
    def punctuate(self):
        # send data to clients every 1 sec
        while True:
            try:
                time_str = time.ctime()
                message = self._protocol.encode_text(time_str, 'text')
                self.send(message)
                time.sleep(1)
            except KeyboardInterrupt:
                break

    def client_sample(self):
        '''
            this function is the core algorithm of testing client sampling schemes
        '''
        scheme = 'random'
        pass

    # boost the server
    def run(self, server_host=SERVER_HOST, server_port=SERVER_PORT):
        # bind and listen
        self._server.bind((server_host, server_port))
        self._server.listen(5)
        logger.info("Server is listening on %s:%s", server_host, server_port)

        # main loop
        while True:
            # get new client
            self.add_client()

            # # punctuate clients
            # if not self._punctuating:
            #     self._punctuating = True
            #     threading.Thread(target=self.punctuate).start()

            # receive data from clients
            # TODO: add aggregation operation, then update model parameters, then send parameter to clients
            for client in self._clients:
                self.handle_client(client)

        # close server
        self.close()