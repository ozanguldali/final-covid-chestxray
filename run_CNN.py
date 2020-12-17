import sys

from cnn.dataset import inv_normalize_tensor
from cnn.helper import set_dataset_and_loaders
from cnn.model import run_model, weighted_model
from cnn.test import test_model

from util.garbage_util import collect_garbage
from util.logger_util import log
from util.tensorboard_util import writer


def main(save=False, dataset_folder="dataset", batch_size=20, img_size=112, test_without_train=False, pretrain_file=None,
         num_workers=4, model_name='alexnet', optimizer_name='Adam', is_pre_trained=False, fine_tune=False,
         num_epochs=18, update_lr=True, normalize=None, validation_freq=0.1, lr=0.001, momentum=0.9, partial=0.125,
         betas=(0.9, 0.99), weight_decay=0.025):
    if not is_pre_trained and fine_tune:
        fine_tune = False

    if test_without_train and pretrain_file is None:
        log.fatal("Pretrained weight file is a must on test without train approach")
        sys.exit(1)

    log.info("Constructing datasets and loaders")
    train_data, train_loader, test_data, test_loader = set_dataset_and_loaders(dataset_folder, batch_size,
                                                                               img_size, num_workers, normalize)

    set_0, set_1 = 0, 0
    for imgs, labels in test_loader:
        if set_0 == 3 and set_1 == 3:
            break

        for e, label in enumerate(labels.tolist()):
            if label == 0 and set_0 != 3:
                writer.add_image("{} - class image sample {}".format(train_data.classes[0], set_0),
                                 inv_normalize_tensor(imgs[e], normalize))
                set_0 += 1
            elif label == 1 and set_1 != 3:
                writer.add_image("{} - class image sample {}".format(train_data.classes[1], set_1),
                                 inv_normalize_tensor(imgs[e], normalize))
                set_1 += 1

            if set_0 == 3 and set_1 == 3:
                break

    log.info("Calling the model: " + model_name)
    if test_without_train:
        model = weighted_model(model_name, pretrain_file)
        test_model(model, test_loader, 0)

    else:
        run_model(model_name=model_name, optimizer_name=optimizer_name, is_pre_trained=is_pre_trained,
                  fine_tune=fine_tune, train_loader=train_loader, test_loader=test_loader,
                  num_epochs=num_epochs, save=save,
                  update_lr=update_lr, dataset_folder=dataset_folder, validation_freq=validation_freq, lr=lr,
                  momentum=momentum, partial=partial, betas=betas, weight_decay=weight_decay)

    collect_garbage()
    writer.close()


if __name__ == '__main__':
    save = False
    log.info("Process Started")
    main(model_name="resnet18", is_pre_trained=True, pretrain_file="84.35_PreTrained_resnet18_Adam_dataset_out",
         img_size=112, test_without_train=True)
    log.info("Process Finished")
