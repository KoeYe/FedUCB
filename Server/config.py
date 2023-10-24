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
configurations = {
    RunMode.DEFAULT: {
        "NAME": "Server",
        "LOGLEVEL": logging.INFO,
        "LOGOUTPUT": OUTPUT.BOTH,
        "LOGFILE": "server.log",
        "CLEAR_LOGFILE": False,
        "SERVER_HOST": "localhost",
        "SERVER_PORT": 6060
    },
    RunMode.DEBUG: {
        "NAME": "Server",
        "LOGLEVEL": logging.DEBUG,
        "LOGOUTPUT": OUTPUT.BOTH,
        "LOGFILE": "server_debug.log",
        "CLEAR_LOGFILE": True,
        "SERVER_HOST": "localhost",
        "SERVER_PORT": 6060
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
    LOGFILE = config["LOGFILE"]
    CLEAR_LOGFILE = config["CLEAR_LOGFILE"]
    SERVER_HOST = config["SERVER_HOST"]
    SERVER_PORT = config["SERVER_PORT"]
    NAME = config["NAME"]

    if CLEAR_LOGFILE and os.path.exists(LOGFILE):
        with open(LOGFILE, "w") as f:
            f.write("")
except KeyError:
    raise Exception("Unknown run mode")
# ---------------------------------------------------------------#

# set the logger ------------------------------------------------#
logger = logging.Logger(NAME)
logger.setLevel(LOGLEVEL)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s - %(filename)s - %(lineno)d")
file_handler = logging.FileHandler(LOGFILE)
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
