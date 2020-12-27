import sys

from torch import optim, nn

from ae import device
from ae.autoencoder import autoencoder
from ae.train import train_model
from ae.save import save_model

from cnn.summary import get_summary

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(train_loader, num_epochs, optimizer_name, lr, weight_decay, momentum, save=False):
    collect_garbage()

    model = autoencoder()

    log.info("Setting the model to device")
    model = model.to(device)

    log.info("The summary:")
    get_summary(model, train_loader)

    collect_garbage()

    log.info("Setting the loss function")
    metric = nn.MSELoss()

    model_parameters = model.parameters()

    if optimizer_name == optim.Adam.__name__:
        optimizer = optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == optim.AdamW.__name__:
        optimizer = optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == optim.SGD.__name__:
        optimizer = optim.SGD(model_parameters, lr=lr, momentum=momentum)
    else:
        log.fatal("not implemented optimizer name: {}".format(optimizer_name))
        sys.exit(1)

    train_model(model, train_loader, metric, optimizer, num_epochs)

    if save:
        save_model(model=model, filename="AE_" + optimizer_name + "_out.pth")
