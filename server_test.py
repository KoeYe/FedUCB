import threading

from Server import Server

if __name__ == "__main__":
    server = Server()
    threading.Thread(server.run()).start()
    server.send("Hello, client!")
    server.close()