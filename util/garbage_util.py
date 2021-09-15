import gc
import torch


def clear_cuda():
    torch.cuda.empty_cache()


def clear_cpu():
    gc.collect()


def collect_garbage():
    clear_cpu()
    clear_cuda()
