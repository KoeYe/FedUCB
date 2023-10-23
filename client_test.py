import threading

from Client import Client

if __name__ == "__main__":
    client = Client()
    client.run()
    print("Sending message to server...")