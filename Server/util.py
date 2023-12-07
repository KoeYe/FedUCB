import random
import os
import json
import matplotlib.pyplot as plt
import torch as pt
import logging
from .config import NAME, LOGLEVEL, LOGOUTPUT, LOGFILE_ROUTES, OUTPUT, BANDWIDTH


# set the logger ------------------------------------------------#
logger = logging.Logger(NAME)
logger.setLevel(LOGLEVEL)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s - %(filename)s - %(lineno)d")
file_handler = logging.FileHandler(LOGFILE_ROUTES)
file_handler.setFormatter(file_formatter)

console_formatter = logging.Formatter("%(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)

handlers = {
    OUTPUT.FILE: [file_handler],
    OUTPUT.CONSOLE: [console_handler],
    OUTPUT.BOTH: [file_handler, console_handler]
}

for handler in handlers.get(LOGOUTPUT):
    logger.addHandler(handler)
# ---------------------------------------------------------------#


def read_data(test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    test_data = {}

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    # logger.info("test data: \n" + str(test_data))

    server_test_data = {
        'y': [],
        'x': []
    }
    for client in test_data.keys():
        try:
            a, b = random.sample(range(len(test_data[client]['y'])), 2)
            y1 = test_data[client]['y'][a]
            y2 = test_data[client]['y'][b]
            x1 = test_data[client]['x'][a]
            x2 = test_data[client]['x'][b]
            server_test_data['y'].append(y1)
            server_test_data['y'].append(y2)
            server_test_data['x'].append(x1)
            server_test_data['x'].append(x2)
        except Exception as e:
            print(len(test_data[client]['y']))
            continue

    return server_test_data

def random_sample(clients, bandwidth):
    '''
        this function is the core algorithm of random sampling
    '''
    # randomly select clients
    random.shuffle(clients)
    selected_clients = clients[:bandwidth]
    return selected_clients
