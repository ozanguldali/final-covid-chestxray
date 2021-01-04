import torch
import torch.nn as nn

from cnn import device

__all__ = ['ProposedNet', 'proposednet']

# Ozan Güldali & Cihat Kırankaya


class ProposedNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ProposedNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.LocalResponseNorm(size=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(num_classes, num_classes))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(num_classes * num_classes * 512, num_classes * num_classes * 64)
        )

        self.fc2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(num_classes * num_classes * 64, num_classes * num_classes * 64)
        )

        self.fc3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(num_classes * num_classes * 64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def proposednet(pretrained=False, pretrained_file=None, **kwargs):
    r"""ProposedNet model architecture

    Args:
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param pretrained_file: pth file name
    """
    if pretrained and pretrained_file is None:
        raise RuntimeError("Pretrained Model Weights File must be specified when pretrained model is wished to be used.")
    model = ProposedNet()

    if pretrained:
        map_location = None if torch.cuda.is_available() else device
        model.load_state_dict(torch.load(pretrained_file, map_location=map_location))

    return model
