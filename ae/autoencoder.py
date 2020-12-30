import torch
import torch.nn as nn

__all__ = ['ConvolutionalAE', 'conv_ae']

from ae import device


class AE_2D(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=1024
        )
        self.encoder_output_layer = nn.Linear(
            in_features=1024, out_features=1024
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=1024, out_features=1024
        )
        self.decoder_output_layer = nn.Linear(
            in_features=1024, out_features=input_shape
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = self.relu(activation)
        code = self.encoder_output_layer(activation)
        code = self.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = self.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = self.relu(activation)

        return reconstructed


class ConvolutionalAE(nn.Module):
    def __init__(self):
        super(ConvolutionalAE, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.t_conv1(x))
        x = self.sigmoid(self.t_conv2(x))

        return x


def ae_2d(input_shape, pretrained=False, pretrained_file=None):
    ae = AE_2D(input_shape)
    if pretrained:
        map_location = None if torch.cuda.is_available() else device
        ae.load_state_dict(torch.load(pretrained_file, map_location=map_location))

    return ae


def conv_ae(pretrained=False, pretrained_file=None):
    ae = ConvolutionalAE()
    if pretrained:
        map_location = None if torch.cuda.is_available() else device
        ae.load_state_dict(torch.load(pretrained_file, map_location=map_location))

    return ae
