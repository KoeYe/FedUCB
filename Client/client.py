import os
import socket
import threading
import struct
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
import asyncio

from .model import Model
from .config import RunMode, SERVER_HOST, SERVER_PORT, RUNMODE, TRAIN_DATA_DIR, TEST_DATA_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, DEVICE, RESULT_DIR, LOCAL_EPOCHS
from .protocol import Protocol
from .util import read_data, logger, data_preprocess, flatten_to_tensor

class Client:
    '''
        Client类有local和online两种
            local: 不需要端口, 直接本地可以训练
            online: 需要端口, 需要和server端链接
        这里其实可以进一步抽象的，但是懒得改了
    '''
    def __init__(self, attr_mode=RUNMODE):
        # 如果用户输入了字符串，这里定义一个字符串到RunMode的映射
        mode_str_to_mode_struct = {
            "default": RunMode.DEFAULT,
            "debug": RunMode.DEBUG,
            "local": RunMode.LOCAL
        }

        # 如果用户输入了字符串，就按用户的输入来，否则按照RunMode来
        if type(attr_mode) == str:
            mode = mode_str_to_mode_struct[attr_mode]
        else:
            mode = attr_mode

        # 这里按照mode来初始化，定义两种初始化方式online和local
        mode_to_init_func = {
            RunMode.DEBUG: self.init_online,
            RunMode.DEFAULT: self.init_online,
            RunMode.LOCAL: self.init_local
        }

        # 调用初始化函数
        mode_to_init_func[mode]()

    def init_online(self):
        '''
            online的初始化方法，需要端口有链接才能初始化成功
        '''
        print("Initializing...")
        self._client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client.connect((SERVER_HOST, SERVER_PORT))
        self._client.setblocking(False)
        # 这里初始化了model，但是参数要等到server端来初始化
        self._model = Model()
        self._protocol = Protocol()
        # recv()的时候需要阻塞，所以这里设置为True
        self._client.setblocking(True)

    def init_local(self):
        '''
            local的初始化方法，不需要端口，直接本地可以训练
        '''
        self._model = Model()

    def send(self, data):
        # 注意这里的data传入的时候就是已经经过protocol.encode()编码过的
        self._client.send(data)

    def recv(self):
        '''
            这个函数用来接收数据
            与recv_all()配合使用
        '''
        length = self._client.recv(4)
        print("length: ", str(struct.unpack('i', length)[0]))
        data = self._client.recv(struct.unpack('i', length)[0])
        print("data_end: ", data[-5:])
        print(b'<END>')
        type, message = self._protocol.decode(data)
        if type == 'text':
            logger.info("Received data: %s", message)
            return message
        elif type == 'params':
            logger.info("aligning parameters")
            self.align_param(message)
            return message

    def align_param(self, data):
        '''
            这个函数用来将server端传来的参数和本地的参数对齐
            由于server端传来的参数是一个list, 所以这里要遍历
            由于本地的参数是一个dict, 所以这里要用dict的方法
        '''
        # logger.debug("parameters: \n" + str(self._model.state_dict()))
        data_dict = {}
        for i in self._model.state_dict().keys():
            data_dict[i] = data.pop(0)
        self._model.load_state_dict(data_dict)
        # logger.debug("aligned parameters: \n" + str(self._model.state_dict()))

    def run_online(self):
        '''
            online的run方法
        '''
        logger.info("running online...")
        self.recv()
        self.training_online()

    def training_online(self):
        # training and send model gradient to server, then update model parameters
        clients, groups, train_data, test_data = read_data(TRAIN_DATA_DIR, TEST_DATA_DIR)

        # select one client to train
        client_id = clients[random.choice(range(len(clients)))]
        client_test_data = test_data[client_id]

        acc, loss = self._model.test(client_test_data)
        logger.info("initial acc: %f, initial loss: %f", acc, loss)

        client_train_data = train_data[client_id]

        x, y = data_preprocess(client_train_data)

        train_dataset = TensorDataset(x, y)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        loss_func = torch.nn.CrossEntropyLoss()

        optimizer = optim.SGD(self._model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        self._model.train()

        local_epoch = 0
        total_epoch = 0
        # start training
        while True:
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                outputs = self._model(inputs)
                loss = loss_func(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            acc, loss = self._model.test(client_test_data)
            logger.info("total_epoch %d, local_epoch %d, acc: %f, loss: %f",total_epoch, local_epoch, acc, loss)
            local_epoch += 1
            if local_epoch == LOCAL_EPOCHS:
                logger.info("send gradients to server")
                # send model gradients to server
                self.send_grad()
                # receive model parameters from server
                self.recv()
                local_epoch = 0
                total_epoch += 1

    def send_grad(self):
        '''
            这个函数用来将本地的model的grad发送给server
        '''
        grad_values = [param for param in self._model.parameters()]
        message = self._protocol.encode_gradients(grad_values, 'grad')
        self.send(message)
        # logger.debug("type: "+ type)

    def run_local(self):
        '''
            local的run方法
            TODO: 这里训练的逻辑需要抽象出来
        '''
        logger.info("running local...")
        clients, groups, train_data, test_data = read_data(TRAIN_DATA_DIR, TEST_DATA_DIR)

        # select one client to train
        client_id = clients[0]
        client_test_data = test_data[client_id]

        acc, loss = self._model.test(client_test_data)
        logger.info("initial acc: %f, initial loss: %f", acc, loss)

        client_train_data = train_data[client_id]

        x, y = data_preprocess(client_train_data)

        train_dataset = TensorDataset(x, y)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        loss_func = torch.nn.CrossEntropyLoss()

        optimizer = optim.SGD(self._model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        self._model.train()

        losses = []
        accuracies = []
        # start training
        for epoch in range(EPOCHS):
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                outputs = self._model(inputs)
                loss = loss_func(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            acc, loss = self._model.test(client_test_data)
            logger.info("epoch %d, acc: %f, loss: %f", epoch, acc, loss)
            losses.append(loss)
            accuracies.append(acc)

        # start drawing
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.text(1, 1, f'LR: {LEARNING_RATE}\nBatch Size: {BATCH_SIZE}', bbox=dict(facecolor='white', alpha=0.5))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Accuracy', color='orange')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()

        # save the figure
        if os.path.exists(RESULT_DIR) == False:
            os.makedirs(RESULT_DIR)
        plt.savefig(RESULT_DIR + '/' + str(LEARNING_RATE) + '-' + str(BATCH_SIZE) + '.png')

        # plt.show()


    def run(self):
        '''
            run方法
            根据mode来选择run的方法
        '''
        # set the run func dict -----------------------------------------#
        mode_to_run_func = {
            RunMode.DEBUG: Client.run_online,
            RunMode.DEFAULT: Client.run_online,
            RunMode.LOCAL: Client.run_local
        }
        # ---------------------------------------------------------------#
        mode_to_run_func[RUNMODE](self)


    def close(self):
        self._client.close()