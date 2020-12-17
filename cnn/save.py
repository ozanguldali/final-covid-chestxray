import torch

from cnn import ROOT_DIR


def save_model(model, path):
    torch.save(model.state_dict(), ROOT_DIR + "/saved_models/" + path)
