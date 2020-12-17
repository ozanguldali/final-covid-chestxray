from torch import nn
from tqdm import tqdm

from cnn import device


def extract_features(data_loader, feature_extractor):
    X, y = [], []

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        X.extend(
            [features.tolist() for features in feature_extractor(images)]
        )
        y.extend(labels.tolist())

    return X, y


# Deep Features from FC2
def alexnet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *[model.classifier[i] for i in range(5)]
    )

    return feature_extractor


# Deep Features from Convolution Base
def resnet_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        nn.Flatten(),
    )

    return feature_extractor


# Deep Features from FC2
def vgg_feature_extractor(model):
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *[model.classifier[i] for i in range(4)]
    )

    return feature_extractor


# Deep Features from Convolution Base
def densenet_feature_extractor(model):
    print(model.classifier.in_features)
    feature_extractor = nn.Sequential(
        model.features,
        nn.ReLU(inplace=True),
        nn.AdaptiveMaxPool2d(1),
        nn.Flatten(),
    )

    return feature_extractor
