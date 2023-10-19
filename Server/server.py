import socket
import threading

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
        self._clients = []

    def add_client(self):
        try:
            client, address = self._server.accept()
            self._clients.append(client)
            print("New client connected:", address)
            print("Current clients number:", len(self._clients))
        except BlockingIOError:
            pass
        except Exception as e:
            print(e)

    def remove_client(self, client):
        print("Client disconnected")
        self._clients.remove(client)
        print("Current clients number:", len(self._clients))

    def close(self):
        print("Closing server...")
        for client in self._clients:
            client.close()
        self._server.close()
        print("Server closed")

    def input_thread(self):
        while True:
            try:
                command = input("Enter command: ")
                if command == "quit":
                    self.close()
                    break
                elif command == "send":
                    self.send()
            except KeyboardInterrupt:
                self.close()
                break

    def send(self):
        with threading.Lock():
            data = input("Enter data to send: ")
        for client in self._clients:
            client.send(data)

    def handle_client(self, client):
        try:
            data = client.recv(1024)
            if data:
                with threading.Lock():
                    print("Received from client:", data.decode())
        except BlockingIOError:
            pass
        except Exception as e:
            self.remove_client(client)
            print(e)

    # TODO: add a thread to send data
    def run(self, server_host=SERVER_HOST, server_port=SERVER_PORT):
        # bind and listen
        self._server.bind((server_host, server_port))
        self._server.listen(5)
        print("Server is listening on", server_host, ":", server_port)

        # create a thread to send data
        # send_thread =

        threading.Thread(target=self.input_thread).start()

        # main loop
        while True:
            # get new client
            try:
                self.add_client()
            except KeyboardInterrupt:
                break

            # receive data from clients
            for client in self._clients:
                try:
                    threading.Thread(target=self.handle_client, args=(client,)).start()
                except KeyboardInterrupt:
                    break

        self.close()