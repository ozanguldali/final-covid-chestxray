import sys

from cnn.helper import set_dataset_and_loaders
from ae.model import run_model

from util.garbage_util import collect_garbage
from util.logger_util import log
from util.tensorboard_util import writer


def main(save=False, dataset_folder="dataset", batch_size=64, img_size=224, num_workers=4, optimizer_name='Adam',
         num_epochs=200, normalize=None, lr=0.001, momentum=0.9, weight_decay=1e-4):

    log.info("Constructing datasets and loaders")
    _, train_loader, _, _ = set_dataset_and_loaders(dataset_folder, batch_size, img_size, num_workers, normalize)

    log.info(train_loader.dataset.class_to_idx)

    log.info("Calling the model AutoEncoder")

    run_model(optimizer_name=optimizer_name, train_loader=train_loader, num_epochs=num_epochs, save=save,
              lr=lr, momentum=momentum, weight_decay=weight_decay)

    collect_garbage()
    writer.close()


if __name__ == '__main__':
    save = False
    log.info("Process Started")
    main(num_epochs=20, save=True)
    log.info("Process Finished")
