import torch

from cnn import device


def load_model(model, path):
    map_location = None if torch.cuda.is_available() else device
    model.load_state_dict(torch.load(path, map_location=map_location))
    model.eval()

    return model


if __name__ == '__main__':
    from torchvision import models
    load_model(models.alexnet(), "")
