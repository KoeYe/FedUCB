import threading

from Client import Client

if __name__ == "__main__":
    client = Client()
    threading.Thread(client.run()).start()
    client.send("Hello, server!")
    print("Sending message to server...")