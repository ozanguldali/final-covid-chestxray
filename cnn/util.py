import sys

from torch import nn
from torchvision import models

from cnn import MODEL_NAME
from cnn.models import proposednet

from util.logger_util import log


def prepare_proposednet(is_pre_trained, fine_tune, num_classes, pretrain_file=None):
    model = proposednet.proposednet(pretrained=is_pre_trained, pretrained_file=pretrain_file, num_classes=num_classes)

    if fine_tune:
        frozen = model.features
        set_parameter_requires_grad(frozen)

    return model


def prepare_alexnet(is_pre_trained, fine_tune, num_classes):
    model = models.alexnet(pretrained=is_pre_trained,
                           num_classes=1000 if is_pre_trained else num_classes)
    if fine_tune:
        frozen = nn.Sequential(
            *[model.features[i] for i in range(3)]
        )
        set_parameter_requires_grad(frozen)

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model


def prepare_resnet(model_name, is_pre_trained, fine_tune, num_classes):
    if model_name == models.resnet50.__name__:
        model = models.resnet50(pretrained=is_pre_trained,
                                num_classes=1000 if is_pre_trained else num_classes)
    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    if fine_tune:
        frozen = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        set_parameter_requires_grad(frozen)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def prepare_squeezenet(model_name, is_pre_trained, fine_tune, num_classes):
    if model_name == models.squeezenet1_0.__name__:
        model = models.squeezenet1_0(pretrained=is_pre_trained,
                                     num_classes=1000 if is_pre_trained else num_classes)
    elif model_name == models.squeezenet1_1.__name__:
        model = models.squeezenet1_1(pretrained=is_pre_trained,
                                     num_classes=1000 if is_pre_trained else num_classes)
    else:
        log.fatal("model name is not known: " + model_name)
        sys.exit(1)

    if fine_tune:
        frozen = nn.Sequential(
            *[model.feeatures[i] for i in range(3)]
        )
        set_parameter_requires_grad(frozen)

    model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels,
                                    num_classes,
                                    kernel_size=model.classifier[1].kernel_size)

    return model


def is_verified(acc):
    model_name = MODEL_NAME[0]

    if model_name == models.alexnet.__name__:
        verified = acc > 87.2

    elif model_name == models.resnet50.__name__:
        verified = acc > 87.2

    elif model_name == models.squeezenet1_1.__name__:
        verified = acc > 87.2

    else:
        verified = True

    return verified


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False
