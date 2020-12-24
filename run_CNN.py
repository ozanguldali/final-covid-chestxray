import sys

from cnn.dataset import inv_normalize_tensor
from cnn.helper import set_dataset_and_loaders
from cnn.model import run_model, weighted_model
from cnn.test import test_model

from util.garbage_util import collect_garbage
from util.logger_util import log
from util.tensorboard_util import writer


def main(save=False, dataset_folder="dataset", batch_size=64, img_size=224, test_without_train=False, pretrain_file=None,
         num_workers=4, model_name='alexnet', optimizer_name='Adam', is_pre_trained=False, fine_tune=False,
         num_epochs=200, update_lr=True, normalize=None, validation_freq=0.05, lr=0.001, momentum=0.9, weight_decay=1e-4):
    if not is_pre_trained and fine_tune:
        fine_tune = False

    if test_without_train and pretrain_file is None:
        log.fatal("Pretrained weight file is a must on test without train approach")
        sys.exit(1)

    log.info("Constructing datasets and loaders")
    train_data, train_loader, test_data, test_loader = set_dataset_and_loaders(dataset_folder, batch_size,
                                                                               img_size, num_workers, normalize)

    log.info(test_loader.dataset.class_to_idx)

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
                  momentum=momentum, weight_decay=weight_decay)

    collect_garbage()
    writer.close()


if __name__ == '__main__':
    save = False
    log.info("Process Started")
    main(model_name="proposedcnn", batch_size=64, num_epochs=8, update_lr=True, validation_freq=0.125)
    log.info("Process Finished")

# alexnet - adam - lr=0.001 - update_lr=True - epochs=60 - acc = 82.64
# alexnet - adamw - lr=0.001 - update_lr=True - epochs=170 - acc = 84.88
# *alexnet - sgd - lr=0.001 - update_lr=False - epochs=30 - acc = 89.53 ------ 2nd

# *resnet18 - adam - lr=0.001 - update_lr=True - epochs=20 - acc = 90.31 ----- 1st
# resnet18 - adamw - lr=0.001 - update_lr=True - epochs=30 - acc = 87.98 ---- 4th
# resnet18 - sgd - lr=0.001 - update_lr=False - epochs=110 - acc = 86.82

# resnet50 - adam - lr=0.001 - update_lr=True - epochs=30 - acc = 83.72
# resnet50 - adamw - lr=0.001 - update_lr=True - epochs=190 - acc = 84.50
# *resnet50 - sgd - lr=0.001 - update_lr=False - epochs=200 - acc = 86.05
# Ör. Test accuracy: 0.8604651162790697
# 2020-12-23 09:18:03,536 - test.py line+42 - INFO - Confusion Matrix:
# [[58  8  0 14]
#  [ 1 78  0  1]
#  [ 0  0 17  1]
#  [ 9  2  0 69]]

# vgg16 - adam - lr=0.001 - update_lr=True - epochs=150 - acc = 86.05
# vgg16 - adamw - lr=0.001 - update_lr=True - epochs=100 - acc = 87.59
# *vgg16 - sgd - lr=0.001 - update_lr=False - epochs=130 - acc = 90.70 ---- 1st
# Ör. Test accuracy: 0.8875968992248062
# 2020-12-23 16:32:04,232 - test.py line+42 - INFO - Confusion Matrix:
# [[61  7  0 12]
#  [ 0 79  0  1]
#  [ 0  0 18  0]
#  [ 5  4  0 71]]
