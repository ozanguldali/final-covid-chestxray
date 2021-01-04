import sys

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from cnn.helper import get_grad_update_params, get_model, get_feature_extractor

from cnn import device, ROOT_DIR, SAVE_FILE, MODEL_NAME
from cnn.models import ensemblenet, proposednet
from cnn.load import load_model
from cnn.save import save_model
from cnn.summary import get_summary
from cnn.test import test_model
from cnn.train import train_model
from cnn.util import is_verified
from util.file_util import path_exists

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(model_name, optimizer_name, is_pre_trained, pretrain_file, fine_tune, num_epochs, train_loader, test_loader,
              validation_freq, lr, momentum, weight_decay, model1_name, model2_name, update_lr=True, save=False):
    collect_garbage()

    MODEL_NAME[0] = model_name

    num_classes = len(train_loader.dataset.classes)

    if model_name == ensemblenet.ensemblenet.__name__:
        model1 = get_model(model_name=model1_name, is_pre_trained=True, fine_tune=False, num_classes=num_classes)
        model2 = get_model(model_name=model2_name, is_pre_trained=True, fine_tune=False, num_classes=num_classes)
        model = ensemblenet.ensemblenet(
            model1=get_feature_extractor(model_name=model1_name, model=model1),
            model2=get_feature_extractor(model_name=model2_name, model=model2),
            in_features=4096 + 196
        )
    else:
        model = get_model(model_name=model_name, is_pre_trained=is_pre_trained, fine_tune=fine_tune,
                          num_classes=num_classes, pretrain_file=pretrain_file)

    log.info("Setting the model to device")
    model = model.to(device)

    log.info("The summary:")
    get_summary(model, train_loader)

    collect_garbage()

    log.info("Setting the loss function")
    metric = nn.CrossEntropyLoss()

    model_parameters = get_grad_update_params(model, fine_tune)

    if optimizer_name == optim.Adam.__name__:
        optimizer = optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == optim.AdamW.__name__:
        optimizer = optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == optim.SGD.__name__:
        optimizer = optim.SGD(model_parameters, lr=lr, momentum=momentum)
    else:
        log.fatal("not implemented optimizer name: {}".format(optimizer_name))
        sys.exit(1)

    log.info("Setting the optimizer as: {}".format(optimizer_name))

    SAVE_FILE[0] = ("" if not is_pre_trained else "PreTrained_") + model_name + "_" + optimizer_name + "_out.pth"

    last_val_iterator = train_model(model, train_loader, test_loader, metric, optimizer, lr=lr,
                                    num_epochs=num_epochs, update_lr=update_lr, validation_freq=validation_freq,
                                    save=save)

    log.info("Testing the model")
    test_acc = test_model(model, test_loader, last_val_iterator)

    if save and is_verified(test_acc):
        exist_files = path_exists(ROOT_DIR + "/saved_models", SAVE_FILE[0], "contains")

        better = len(exist_files) == 0
        if not better:
            exist_acc = []
            for file in exist_files:
                exist_acc.append(float(file.split("_")[0].replace(",", ".")))
            better = all(test_acc > acc for acc in exist_acc)
        if better:
            save_model(model=model, filename=str(round(test_acc, 2)) + "_" + SAVE_FILE[0])

    return model


def weighted_model(model_name, pretrain_file, use_actual_num_classes=False):
    out_file = ROOT_DIR + "/saved_models/" + pretrain_file + ".pth"

    if model_name == proposednet.proposednet.__name__:
        model = proposednet.proposednet(pretrained=True, pretrained_file=out_file)
        return model

    else:

        if model_name == models.alexnet.__name__:
            model = models.alexnet(num_classes=4 if use_actual_num_classes else 1000)

        elif model_name == models.resnet50.__name__:
            model = models.resnet50(num_classes=4 if use_actual_num_classes else 1000)

        elif model_name == models.squeezenet1_1.__name__:
            model = models.squeezenet1_1(num_classes=4 if use_actual_num_classes else 1000)

        else:
            log.fatal("model name is not known: " + model_name)
            sys.exit(1)

        try:
            log.info("Using class size as: {}".format(4 if use_actual_num_classes else 1000))
            return load_model(model, out_file)
        except RuntimeError as re:
            log.error(re)
            return weighted_model(model_name, pretrain_file, True)
