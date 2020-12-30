import sys

from cnn.dataset import set_loader
from cnn.helper import set_dataset_and_loaders
from ae.model import run_model

from util.garbage_util import collect_garbage
from util.logger_util import log
from util.tensorboard_util import writer


def main(save=False, dataset_folder="dataset", batch_size=64, img_size=224, num_workers=4, optimizer_name='Adam',
         num_epochs=200, normalize=None, lr=0.001, momentum=0.9, weight_decay=1e-4):

    log.info("Constructing datasets and loaders")
    train_data, _, test_data, _ = set_dataset_and_loaders(dataset_folder, batch_size, img_size, num_workers, normalize)

    log.info("Merging CNN train&test datasets")
    dataset = train_data + test_data

    log.info("Constructing loader for merged dataset")
    data_loader = set_loader(dataset=dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)

    log.info((train_data+test_data).datasets[1].class_to_idx)

    log.info("Calling the model AutoEncoder")

    run_model(optimizer_name=optimizer_name, train_loader=data_loader, num_epochs=num_epochs, save=save,
              lr=lr, momentum=momentum, weight_decay=weight_decay)

    collect_garbage()
    writer.close()


if __name__ == '__main__':
    save = False
    log.info("Process Started")
    main(num_epochs=2, save=True)
    log.info("Process Finished")
