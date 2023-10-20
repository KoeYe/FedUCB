import socket
import threading
import time

from .configure import SERVER_HOST
from .configure import SERVER_PORT

#
# Server class
#   for simple sake, this server just listen to one port
#
class Server:
    def __init__(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setblocking(False)
        self._punctuating = False
        self._client_num = 0
        self._clients = []

    def add_client(self):
        try:
            client, address = self._server.accept()
            self._clients.append(client)
            self._client_num += 1
            print("New client connected:", address)
            print("Current clients number:", self._client_num)
        except BlockingIOError:
            pass
        except Exception as e:
            print(e)

    def remove_client(self, client):
        print("Client disconnected")
        self._clients.remove(client)
        self._client_num -= 1
        print("Current clients number:", self._client_num)

    def close(self):
        print("Closing server...")
        for client in self._clients:
            client.close()
        self._server.close()
        print("Server closed")

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
                print("Received from client:", data.decode())
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
        print("Server is listening on", server_host, ":", server_port)

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
            for client in self._clients:
                self.handle_client(client)

        # close server
        self.close()