import torch
import torch.nn as nn

__all__ = ['EnsembleNet', 'ensemblenet']


class EnsembleNet(nn.Module):
    def __init__(self, model1, model2, in_features, num_classes=4):
        super(EnsembleNet, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x1 = self.model1(x.clone())
        x1 = x1.view(x1.size(0), -1)

        x2 = self.model2(x)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)

        return x


def ensemblenet(model1, model2, in_features, **kwargs):
    r"""EnsembleNet model from 2 CNN models
    """
    model = EnsembleNet(model1, model2, in_features)
    return model
