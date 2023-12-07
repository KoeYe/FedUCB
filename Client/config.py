import logging
import os
from enum import Enum

# define the run mode of the server ------------------------------#
class RunMode(Enum):
    DEFAULT = 1
    DEBUG = 2
    LOCAL = 3
# ----------------------------------------------------------------#


# define the output of the logger --------------------------------#
class OUTPUT(Enum):
    FILE = 1
    CONSOLE = 2
    BOTH = 3
# ---------------------------------------------------------------#

# set the configurations ----------------------------------------#
# here are fixed configurations
NAME = "Client" # log name not necessary
LOG_DIR = "logs" # dir of logs
TRAIN_DATA_DIR = "./data/femnist/train"
TEST_DATA_DIR = "./data/femnist/test"

# here are changeable configurations
configurations = {
    RunMode.DEFAULT: {
        "LOGLEVEL": logging.INFO,
        "LOGOUTPUT": OUTPUT.BOTH,
        "LOGFILE_NAME": "client.log",
        "CLEAR_LOGFILE": True,
        "SERVER_HOST": "localhost",
        "SERVER_PORT": 6060,
        "EPOCHS": 100,
        "BATCH_SIZE": 10,
        "LEARNING_RATE": 0.0001,
        "MOMENTUM": 0.9,
        "LOCAL_EPOCHS": 5,
        "DEVICE": "cpu",
        "RESULT_DIR": "./result",
    },
    RunMode.DEBUG: {
        "LOGLEVEL": logging.DEBUG,
        "LOGOUTPUT": OUTPUT.BOTH,
        "LOGFILE_NAME": "client_debug.log",
        "CLEAR_LOGFILE": True,
        "SERVER_HOST": "localhost",
        "SERVER_PORT": 6060,
        "EPOCHS": 100,
        "BATCH_SIZE": 10,
        "LEARNING_RATE": 0.0001,
        "MOMENTUM": 0.9,
        "LOCAL_EPOCHS": 5,
        "DEVICE": "cpu",
        "RESULT_DIR": "./result_debug",
    },
    RunMode.LOCAL: {
        "LOGLEVEL": logging.DEBUG,
        "LOGOUTPUT": OUTPUT.BOTH,
        "LOGFILE_NAME": "client_local.log",
        "CLEAR_LOGFILE": True,
        "SERVER_HOST": "localhost",
        "SERVER_PORT": 6060,
        "EPOCHS": 500,
        "BATCH_SIZE": 10,
        "LEARNING_RATE": 0.0001,
        "MOMENTUM": 0.9,
        "DEVICE": "cpu",
        "RESULT_DIR": "./result_local"
    }
}
# ---------------------------------------------------------------#

# set the run mode ----------------------------------------------#
RUNMODE = RunMode.DEBUG
# ---------------------------------------------------------------#

# get the configurations ----------------------------------------#
try:
    config = configurations[RUNMODE]
    LOGLEVEL = config["LOGLEVEL"]
    LOGOUTPUT = config["LOGOUTPUT"]
    LOGFILE_NAME = config["LOGFILE_NAME"]
    CLEAR_LOGFILE = config["CLEAR_LOGFILE"]
    SERVER_HOST = config["SERVER_HOST"]
    SERVER_PORT = config["SERVER_PORT"]
    LOGFILE_ROUTES = LOG_DIR + "/" + LOGFILE_NAME
    EPOCHS = config["EPOCHS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    MOMENTUM = config["MOMENTUM"]
    DEVICE = config["DEVICE"]
    LOCAL_EPOCHS = config["LOCAL_EPOCHS"]
    RESULT_DIR = config["RESULT_DIR"]


    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    if CLEAR_LOGFILE and os.path.exists(LOGFILE_ROUTES):
        with open(LOGFILE_ROUTES, "w") as f:
            f.write("")

except KeyError:
    raise Exception("Unknown run mode")
# ---------------------------------------------------------------#
