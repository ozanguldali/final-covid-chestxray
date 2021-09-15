import torch
import torch.nn as nn

from cnn import device

__all__ = ['DarkCovidNet', 'darkcovidnet']

# https://doi.org/10.1016/j.compbiomed.2020.103792
# https://www.researchgate.net/publication/340935440_Automated_Detection_of_COVID-19_Cases_Using_Deep_Neural_Networks_with_X-ray_Images
# https://github.com/muhammedtalo/COVID-19


class DarkCovidNet(nn.Module):
    def __init__(self, num_classes=4):
        super(DarkCovidNet, self).__init__()

        self.features = nn.Sequential(
                            self.conv_block(3, 8),
                            self.maxpooling(),
                            self.conv_block(8, 16),
                            self.maxpooling(),
                            self.triple_conv(16, 32),
                            self.maxpooling(),
                            self.triple_conv(32, 64),
                            self.maxpooling(),
                            self.triple_conv(64, 128),
                            self.maxpooling(),
                            self.triple_conv(128, 256),
                            self.conv_block(256, 128, size=1),
                            self.conv_block(128, 256),
                            self.conv_layer(256, num_classes)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(13, 13))
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(13 * 13 * num_classes, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def conv_layer(selfself, ni, nf, size=3, stride=1):
        for_pad = lambda s: s if s > 2 else 3
        return nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=size, stride=stride,
                      padding=(for_pad(size) - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nf),
        )

    def conv_block(self, ni, nf, size=3, stride=1):
        for_pad = lambda s: s if s > 2 else 3
        return nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=size, stride=stride,
                      padding=(for_pad(size) - 1) // 2, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def triple_conv(self, ni, nf):
        return nn.Sequential(
            self.conv_block(ni, nf),
            self.conv_block(nf, ni, size=1),
            self.conv_block(ni, nf)
        )

    def maxpooling(self):
        return nn.MaxPool2d(2, stride=2)


def darkcovidnet(pretrained=False, pretrained_file=None, **kwargs):
    r"""DarkCovidNet model architecture

    Args:
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param pretrained_file: pth file name
    """
    if pretrained and pretrained_file is None:
        raise RuntimeError("Pretrained Model Weights File must be specified when pretrained model is wished to be used.")
    model = DarkCovidNet()

    if pretrained:
        map_location = None if torch.cuda.is_available() else device
        model.load_state_dict(torch.load(pretrained_file, map_location=map_location))

    return model
