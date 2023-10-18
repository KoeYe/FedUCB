import socket

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

    def add_client(self, client):
        self._clients.append(client)

    def remove_client(self, client):
        self._clients.remove(client)

    def send(self, data):
        for client in self._clients:
            client.send(data)

    def run(self, server_host=SERVER_HOST, server_port=SERVER_PORT):
        self._server.bind((server_host, server_port))
        self._server.listen(5)
        print("Server is listening on", server_host, ":", server_port)
        while True:
            try:
                client, address = self._server.accept()
                print("Client connected from", address)
                self.add_client(client)
            except BlockingIOError:
                pass
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)

            for client in self._clients:
                try:
                    data = client.recv(1024)
                    if data:
                        print("Received from client:", data.decode())
                        self.send(data)
                    else:
                        print("Client disconnected")
                        self.remove_client(client)
                except BlockingIOError:
                    pass
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(e)

        self._server.close()