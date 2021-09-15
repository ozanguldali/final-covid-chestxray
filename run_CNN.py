import sys

from cnn import device
from cnn.helper import set_dataset_and_loaders
from cnn.model import run_model, weighted_model
from cnn.test import test_model

from util.garbage_util import collect_garbage
from util.logger_util import log
from util.tensorboard_util import writer


def main(save=False, dataset_folder="dataset", batch_size=64, img_size=224, test_without_train=False, pretrain_file=None,
         num_workers=4, model_name='alexnet', optimizer_name='Adam', is_pre_trained=False, fine_tune=False,
         model1_name="", model2_name="", num_epochs=200, update_lr=True, normalize=None, validation_freq=0.05,
         lr=0.001, momentum=0.9, weight_decay=1e-4):

    if test_without_train and pretrain_file is None:
        log.fatal("Pretrained weight file is a must on test without train approach.")
        sys.exit(1)

    if not is_pre_trained and fine_tune:
        fine_tune = False

    if model_name == "ensemblenet" and (model1_name == "" or model2_name == ""):
        log.fatal("Two models must be specified to create an ensemble cnn model.")

    log.info("Constructing datasets and loaders")
    train_data, train_loader, test_data, test_loader = set_dataset_and_loaders(dataset_folder, batch_size,
                                                                               img_size, num_workers, normalize)

    log.info(test_loader.dataset.class_to_idx)

    log.info("Calling the model: " + model_name)
    if test_without_train:
        model = weighted_model(model_name, pretrain_file)
        model = model.to(device)
        test_model(model, test_loader, 0)

    else:
        run_model(model_name=model_name, optimizer_name=optimizer_name, is_pre_trained=is_pre_trained,
                  pretrain_file=pretrain_file, fine_tune=fine_tune, train_loader=train_loader, test_loader=test_loader,
                  num_epochs=num_epochs, save=save, model1_name=model1_name, model2_name=model2_name,
                  update_lr=update_lr, validation_freq=validation_freq, lr=lr,
                  momentum=momentum, weight_decay=weight_decay)

    collect_garbage()
    writer.close()


if __name__ == '__main__':
    save = False
    log.info("Process Started")
    main(model_name="proposednet", test_without_train=True, pretrain_file="87.21_proposednet_AdamW_out")
    log.info("Process Finished")


# proposed - pretrained = False - adam  - lr=0.0001 - update_lr=True  - epochs=40 - acc = 87.21
# proposed - pretrained = False - adamw - lr=0.0001 - update_lr=True  - epochs=120 - acc = 87.21
# proposed - pretrained = False - sgd   - lr=0.001 - update_lr=False - epochs=190 - acc = 87.21

# darkcovidnet - pretrained = False - adam  - lr=0.001 - update_lr=True  - epochs=170 - acc = 84.88
# darkcovidnet - pretrained = False - adamw - lr=0.001 - update_lr=True  - epochs=40 - acc = 86.04
# darkcovidnet - pretrained = False - sgd   - lr=0.001 - update_lr=False - epochs=170 - acc = 83.33

# aNovelNet - pretrained = False - adam  - lr=0.001 - update_lr=True  - epochs=20 - acc = 79.84
# aNovelNet - pretrained = False - adamw - lr=0.001 - update_lr=True  - epochs=40 - acc = 81.00
# aNovelNet - pretrained = False - sgd   - lr=0.001 - update_lr=False - epochs=50 - acc = 82.55

# covidnet - pretrained = False - adam  - lr=0.0001 - update_lr=True  - epochs=100 - acc = 86.04
# covidnet - pretrained = False - adamw - lr=0.0001 - update_lr=True  - epochs=70 - acc = 84.88
# covidnet - pretrained = False - sgd   - lr=0.001 - update_lr=False - epochs=80 - acc = 82.55

# resnet50 - adam  - lr=0.001 - update_lr=True  - epochs=30 - acc = 83.72
# resnet50 - adamw - lr=0.001 - update_lr=True  - epochs=190 - acc = 84.49
# resnet50 - sgd   - lr=0.001 - update_lr=False - epochs=200 - acc = 86.04

# alexnet - adam  - lr=0.001 - update_lr=True  - epochs=60 - acc = 82.64
# alexnet - adamw - lr=0.001 - update_lr=True  - epochs=170 - acc = 84.88
# alexnet - sgd   - lr=0.01 - update_lr=False - epochs=30 - acc = 85.27

# squeezenet1_1 - adam  - lr=0.0001 - update_lr=True  - epochs=130 - acc = 83.72
# squeezenet1_1 - adamw - lr=0.0001 - update_lr=True  - epochs=110 - acc = 84.88
# squeezenet1_1 - sgd   - lr=0.0001 - update_lr=False - epochs=200 - acc = 75.88

# Ex. Test accuracy: 0.8488372093023255
# 2020-12-25 17:46:07,936 - test.py line+42 - INFO - Confusion Matrix:
# [[59  6  1 14]
#  [ 1 79  0  0]
#  [ 1  0 16  1]
#  [12  3  0 65]]

# Ex. Test accuracy: 0.8604651162790697
# 2020-12-23 09:18:03,536 - test.py line+42 - INFO - Confusion Matrix:
# [[58  8  0 14]
#  [ 1 78  0  1]
#  [ 0  0 17  1]
#  [ 9  2  0 69]]

# Ex. Test accuracy: 0.872093023255814
# 2020-12-26 01:01:14,550 - test.py line+42 - INFO - Confusion Matrix:
# [[68  4  1  7]
#  [ 1 75  4  0]
#  [ 0  0 17  1]
#  [12  3  0 65]]

# Ex. 2020-12-27 21:28:19,780 - test.py line+37 - INFO -
# Test accuracy: 0.872093023255814
# 2020-12-27 21:28:19,787 - test.py line+42 - INFO - Confusion Matrix:
# [[61  6  1 12]
#  [ 1 77  1  1]
#  [ 0  0 17  1]
#  [ 6  4  0 70]]
# 2020-12-27 21:28:19,796 - test.py line+45 - INFO - Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.90      0.76      0.82        80
#            1       0.89      0.96      0.92        80
#            2       0.89      0.94      0.92        18
#            3       0.83      0.88      0.85        80
#
#     accuracy                           0.87       258
#    macro avg       0.88      0.89      0.88       258
# weighted avg       0.87      0.87      0.87       258
