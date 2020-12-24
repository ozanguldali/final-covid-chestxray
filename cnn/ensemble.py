import torch
import torch.nn as nn
from pytools import F

from cnn import device

__all__ = ['EnsembleNet', 'ensemblenet']


class EnsembleNet(nn.Module):
    def __init__(self, model1, model2, num_classes=4):
        super(EnsembleNet, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.classifier = nn.Linear(2 * num_classes, num_classes)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model1(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)

        return x


def ensemblenet(model1, model2, **kwargs):
    r"""EnsembleNet model from 2 CNN models
    """
    model = EnsembleNet(model1, model2)
    return model
