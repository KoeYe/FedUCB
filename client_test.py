from Client import Client

if __name__ == "__main__":
    client = Client()
    while True:
        try:
            data = input("Enter data to send: ")
            client.send(data)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
    client.close()