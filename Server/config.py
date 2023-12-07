import logging
import os
from enum import Enum

# define the run mode of the server ------------------------------#
class RunMode(Enum):
    DEFAULT = 1
    DEBUG = 2
# ----------------------------------------------------------------#

# define the output of the logger --------------------------------#
class OUTPUT(Enum):
    FILE = 1
    CONSOLE = 2
    BOTH = 3
# ---------------------------------------------------------------#

# set the configurations ----------------------------------------#
NAME = "Server"
LOG_DIR = "logs"
BANDWIDTH = 10
TEST_DATA_DIR = "./data/femnist/test"

configurations = {
    RunMode.DEFAULT: {
        "NAME": NAME,
        "LOGLEVEL": logging.INFO,
        "LOGOUTPUT": OUTPUT.BOTH,
        "LOGFILE": "server.log",
        "CLEAR_LOGFILE": False,
        "SERVER_HOST": "localhost",
        "SERVER_PORT": 6060,
        "LEARNING_RATE": 0.0001,
        "MOMENTUM": 0.9,
        "BATCH_SIZE": 10,
        "EPOCHS": 100,
        "DEVICE": "cpu",
        "BANDWIDTH": BANDWIDTH,
        "TEST_DATA_DIR": TEST_DATA_DIR
    },
    RunMode.DEBUG: {
        "NAME": NAME,
        "LOGLEVEL": logging.DEBUG,
        "LOGOUTPUT": OUTPUT.BOTH,
        "LOGFILE_NAME": "server_debug.log",
        "CLEAR_LOGFILE": True,
        "SERVER_HOST": "localhost",
        "SERVER_PORT": 6060,
        "LEARNING_RATE": 0.0001,
        "MOMENTUM": 0.9,
        "BATCH_SIZE": 10,
        "EPOCHS": 100,
        "DEVICE": "cpu",
        "BANDWIDTH": BANDWIDTH,
        "TEST_DATA_DIR": TEST_DATA_DIR
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
    NAME = config["NAME"]
    LOGFILE_ROUTES = LOG_DIR + "/" + LOGFILE_NAME
    BANDWIDTH = config["BANDWIDTH"]
    LEARNING_RATE = config["LEARNING_RATE"]
    MOMENTUM = config["MOMENTUM"]
    BATCH_SIZE = config["BATCH_SIZE"]
    EPOCHS = config["EPOCHS"]
    DEVICE = config["DEVICE"]
    TEST_DATA_DIR = config["TEST_DATA_DIR"]

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    if CLEAR_LOGFILE and os.path.exists(LOGFILE_ROUTES):
        print("Clearing log file...")
        with open(LOGFILE_ROUTES, "w") as f:
            f.write("")
except KeyError:
    raise Exception("Unknown run mode")
# ---------------------------------------------------------------#
