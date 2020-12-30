import os

import torch

from util.logger_util import log
from ae.autoencoder import conv_ae

ROOT_DIR = str(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device is selected as %s" % device)
SAVE_FILE = [""]
MODEL_NAME = [""]

ae = conv_ae(True, ROOT_DIR.replace("/cnn", "") + "/ae/saved_aes/AE_Adam_out.pth").to(device=device)

