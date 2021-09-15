import os

import torch

from util.logger_util import log

ROOT_DIR = str(os.path.dirname(os.path.abspath(__file__)))


# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# log.info("envion is set as: %s" % str(os.environ["KERAS_BACKEND"]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device is selected as %s" % device)
SAVE_FILE = [""]
MODEL_NAME = [""]

