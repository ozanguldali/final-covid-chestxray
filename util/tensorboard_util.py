import os

# from util.file_util import prepare_directory
from torch.utils.tensorboard import SummaryWriter

root_dir = str(os.path.dirname(os.path.abspath(__file__))).split("util")[0]
log_dir = root_dir + "tensorboard-logs/"
# prepare_directory(log_dir)
writer = SummaryWriter(log_dir=log_dir)
