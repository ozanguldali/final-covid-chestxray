import os
import sys

import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, Normalizer

import run_CNN
import run_ML
from cnn import model as cnn_model, device
from cnn.dataset import set_loader
from cnn.features import extract_features
from cnn.helper import set_dataset_and_loaders, get_feature_extractor
from ml import model as ml_model
from util.garbage_util import collect_garbage
from util.logger_util import log

ROOT_DIR = str(os.path.dirname(os.path.abspath(__file__)))


def main(transfer_learning, load_numpy=False, method="", ml_model_name="svm", cv=10, dataset_folder="dataset",
         pretrain_file=None, batch_size=32, img_size=224, num_workers=4, cnn_model_name="", optimizer_name='Adam',
         validation_freq=0.1, lr=0.001, momentum=0.9, weight_decay=1e-4,
         update_lr=True, is_pre_trained=False, fine_tune=False, num_epochs=16, normalize=True, lambdas=None, seed=4):

    if lambdas is None:
        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    if not transfer_learning:
        if method.lower() == "ml":
            run_ML.main(model_name=ml_model_name, dataset_folder=dataset_folder, seed=seed, cv=cv,
                        img_size=img_size, normalize=normalize, lambdas=lambdas)
        elif method.lower() == "cnn":
            run_CNN.main(save=False, dataset_folder=dataset_folder, batch_size=batch_size, test_without_train=False,
                         img_size=img_size, num_workers=num_workers, num_epochs=num_epochs, model_name=cnn_model_name,
                         optimizer_name=optimizer_name, is_pre_trained=is_pre_trained, fine_tune=fine_tune,
                         update_lr=update_lr, normalize=normalize, validation_freq=validation_freq, lr=lr,
                         momentum=momentum, weight_decay=weight_decay)
        else:
            log.fatal("method name is not known: " + method)
            sys.exit(1)

    else:
        log.info("Constructing datasets and loaders")
        train_data, train_loader, test_data, test_loader = set_dataset_and_loaders(dataset_folder=dataset_folder,
                                                                                   batch_size=batch_size,
                                                                                   img_size=img_size,
                                                                                   num_workers=num_workers,
                                                                                   normalize=normalize)

        if load_numpy:
            X_cnn = np.load("X_cnn.npy")
            y = list(np.load("y.npy"))

        else:
            if is_pre_trained and pretrain_file is not None and \
                    cnn_model_name in pretrain_file.lower():
                log.info("Getting PreTrained CNN model: " + cnn_model_name + " from the Weights of " + pretrain_file)
                model = cnn_model.weighted_model(cnn_model_name, pretrain_file)

            else:
                log.info("Running CNN model: " + cnn_model_name)
                model = cnn_model.run_model(model_name=cnn_model_name, optimizer_name=optimizer_name, fine_tune=fine_tune,
                                            is_pre_trained=is_pre_trained, train_loader=train_loader, num_epochs=num_epochs,
                                            test_loader=test_loader, validation_freq=validation_freq, lr=lr,
                                            momentum=momentum, weight_decay=weight_decay, pretrain_file=pretrain_file,
                                            update_lr=update_lr, save=False, model1_name="", model2_name="")

            log.info("Feature extractor is being created")
            feature_extractor = get_feature_extractor(cnn_model_name, model.eval())
            log.info("Feature extractor is setting to device: " + str(device))
            feature_extractor = feature_extractor.to(device)

            log.info("Merging CNN train&test datasets")
            dataset = train_data + test_data

            log.info("Constructing loader for merged dataset")
            data_loader = set_loader(dataset=dataset, batch_size=int(len(dataset) / 5), shuffle=False,
                                     num_workers=num_workers)
            log.info("Extracting features as X_cnn array and labels as general y vector")
            X_cnn, y = extract_features(data_loader, feature_extractor)

        class_dist = {i: y.count(i) for i in y}
        class0_size = class_dist[0]
        class1_size = class_dist[1]
        class3_size = class_dist[2]
        class4_size = class_dist[3]
        log.info("Total class 0 size: " + str(class0_size))
        log.info("Total class 1 size: " + str(class1_size))
        log.info("Total class 3 size: " + str(class3_size))
        log.info("Total class 4 size: " + str(class4_size))

        if normalize:
            X_cnn = Normalizer().fit_transform(X_cnn)
        X_cnn = StandardScaler().fit_transform(X_cnn)

        log.info("Number of features in X_cnn: " + str(len(X_cnn[0])))

        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

        ml_model.run_model(model_name=ml_model_name, X=X_cnn, y=y, seed=seed, kf=kf, lambdas=lambdas)

    collect_garbage()


if __name__ == '__main__':
    log.info("Process Started")
    main(transfer_learning=True, load_numpy=True, seed=4)

    log.info("Process Finished")

# C: 5.0 - kernel: rbf - gamma: scale - decision_function_shape: ovo - seed: 4
# 2020-12-28 01:02:16,553 - model.py line+20 - INFO - Running ML model: svm
# 2021-01-04 20:13:27,292 - helper.py line+51 - INFO - 10-Fold CV Average Test Success Ratio: 97.94946484216231%
# 2021-01-04 20:13:27,292 - helper.py line+52 - INFO - 10-Fold CV Average AUC Score: 0.9959950265035653
# 2021-01-04 20:13:27,293 - helper.py line+53 - INFO - 10-Fold CV Average Confusion Matrix:
# [[30.9  0.1  0.1  0.9]
#  [ 0.1 31.7  0.1  0.1]
#  [ 0.   0.  25.7  0.1]
#  [ 0.8  0.2  0.  31. ]]
# 2020-12-28 01:02:24,997 - util.py line+42 - INFO -
# 2020-12-28 01:17:40,682 - util.py line+16 - INFO - Penalty Enabled: True
# Fitting 10 folds for each of 6 candidates, totalling 60 fits
# [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
# [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:   10.9s
# [Parallel(n_jobs=4)]: Done  60 out of  60 | elapsed:   41.8s finished
# 2020-12-28 01:18:23,013 - util.py line+33 - INFO - Best lambda value has determined as: 0.05
# 2020-12-28 01:18:25,944 - helper.py line+48 - INFO - 10-Fold CV Average Test Success Ratio: 97.37298469042136%
# 2020-12-28 01:18:25,944 - helper.py line+49 - INFO - 10-Fold CV Average AUC Score: 0.9940956548428141
# 2020-12-28 01:18:25,945 - helper.py line+50 - INFO - 10-Fold CV Average Confusion Matrix:
# [[30.2  0.6  0.1  1.1]
#  [ 0.2 31.6  0.1  0.1]
#  [ 0.   0.  25.7  0.1]
#  [ 0.7  0.2  0.  31.1]]
# 2020-12-28 01:02:24,997 - util.py line+42 - INFO -
# 2020-12-28 01:02:24,998 - model.py line+23 - INFO - Running ML model: lr
# 2020-12-28 01:02:24,998 - util.py line+51 - INFO - Penalty Enabled: False
# 2020-12-28 01:02:36,754 - helper.py line+48 - INFO - 10-Fold CV Average Test Success Ratio: 97.20701801923856%
# 2020-12-28 01:02:36,754 - helper.py line+49 - INFO - 10-Fold CV Average AUC Score: 0.9940363294023189
# 2020-12-28 01:02:36,754 - helper.py line+50 - INFO - 10-Fold CV Average Confusion Matrix:
# [[30.3  0.3  0.1  1.3]
#  [ 0.1 31.6  0.1  0.2]
#  [ 0.   0.  25.7  0.1]
#  [ 1.   0.2  0.  30.8]]
# 2020-12-28 01:02:36,755 - util.py line+74 - INFO -
# 2020-12-28 01:02:36,819 - app.py line+105 - INFO - Process Finished

############

# /Users/ozanguldali/opt/anaconda3/bin/python /Users/ozanguldali/Documents/master_courses/deep_learning/final_project/final-covid-chestxray/app.py
# 2020-12-23 20:28:03,481 - __init__.py line+14 - INFO - Device is selected as cpu
# 2020-12-23 20:28:07,644 - app.py line+100 - INFO - Process Started
# 2020-12-23 20:28:07,645 - app.py line+43 - INFO - Constructing datasets and loaders
# 2020-12-23 20:28:07,645 - helper.py line+16 - INFO - Setting train data
# 2020-12-23 20:28:07,649 - helper.py line+18 - INFO - Train data length: 960
# 2020-12-23 20:28:07,649 - helper.py line+19 - INFO - Setting test data
# 2020-12-23 20:28:07,650 - helper.py line+21 - INFO - Test data length: 258
# 2020-12-23 20:28:07,650 - helper.py line+23 - INFO - Setting train loader
# 2020-12-23 20:28:07,650 - helper.py line+25 - INFO - Setting test loader
# 2020-12-23 20:28:07,650 - app.py line+52 - INFO - Getting PreTrained CNN model: vgg16 from the Weights of 90.7_PreTrained_vgg16_SGD_out
# 2020-12-23 20:28:08,855 - model.py line+113 - INFO - Using class size as: 1000
# 2020-12-23 20:28:10,215 - model.py line+116 - ERROR - Error(s) in loading state_dict for VGG:
# 	size mismatch for classifier.6.weight: copying a param with shape torch.Size([4, 4096]) from checkpoint, the shape in current model is torch.Size([1000, 4096]).
# 	size mismatch for classifier.6.bias: copying a param with shape torch.Size([4]) from checkpoint, the shape in current model is torch.Size([1000]).
# 2020-12-23 20:28:11,397 - model.py line+113 - INFO - Using class size as: 4
# 2020-12-23 20:28:12,712 - app.py line+63 - INFO - Feature extractor is being created
# 2020-12-23 20:28:12,713 - app.py line+65 - INFO - Feature extractor is setting to device: cpu
# 2020-12-23 20:28:12,713 - app.py line+68 - INFO - Merging CNN train&test datasets
# 2020-12-23 20:28:12,713 - app.py line+71 - INFO - Constructing loader for merged dataset
# 2020-12-23 20:28:12,714 - app.py line+74 - INFO - Extracting features as X_cnn array and labels as general y vector
# 100%|██████████| 6/6 [10:15<00:00, 102.58s/it]
# 2020-12-23 20:38:28,238 - app.py line+81 - INFO - Total class 0 size: 320
# 2020-12-23 20:38:28,248 - app.py line+82 - INFO - Total class 1 size: 320
# 2020-12-23 20:38:28,248 - app.py line+83 - INFO - Total class 3 size: 258
# 2020-12-23 20:38:28,248 - app.py line+84 - INFO - Total class 4 size: 320
# 2020-12-23 20:38:29,176 - app.py line+90 - INFO - Number of features in X_cnn: 4096
# 2020-12-23 20:38:29,775 - model.py line+20 - INFO - Running ML model: svm
# 2020-12-23 20:38:29,775 - util.py line+15 - INFO - Penalty Enabled: False
# 2020-12-23 20:39:49,535 - helper.py line+48 - INFO - 10-Fold CV Average Test Success Ratio: 97.94472293727135%
# 2020-12-23 20:39:49,535 - helper.py line+49 - INFO - 10-Fold CV Average AUC Score: 0.9971283508924493
# 2020-12-23 20:39:49,535 - helper.py line+50 - INFO - 10-Fold CV Average Confusion Matrix:
# [[30.5  0.6  0.   0.9]
#  [ 0.1 31.9  0.   0. ]
#  [ 0.   0.  25.8  0. ]
#  [ 0.6  0.3  0.  31.1]]
# 2020-12-23 20:39:49,537 - util.py line+39 - INFO -
# 2020-12-23 20:39:49,537 - util.py line+15 - INFO - Penalty Enabled: True
# Fitting 10 folds for each of 6 candidates, totalling 60 fits
# [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
# [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:   19.2s
# [Parallel(n_jobs=4)]: Done  60 out of  60 | elapsed:   40.5s finished
# 2020-12-23 20:40:31,321 - util.py line+32 - INFO - Best lambda value has determined as: 0.05
# 2020-12-23 20:40:40,341 - helper.py line+48 - INFO - 10-Fold CV Average Test Success Ratio: 98.2732692047148%
# 2020-12-23 20:40:40,342 - helper.py line+49 - INFO - 10-Fold CV Average AUC Score: 0.9964514392031891
# 2020-12-23 20:40:40,342 - helper.py line+50 - INFO - 10-Fold CV Average Confusion Matrix:
# [[30.8  0.5  0.   0.7]
#  [ 0.  32.   0.   0. ]
#  [ 0.   0.  25.8  0. ]
#  [ 0.6  0.3  0.  31.1]]
# 2020-12-23 20:40:40,345 - util.py line+39 - INFO -
# 2020-12-23 20:40:40,346 - model.py line+23 - INFO - Running ML model: lr
# 2020-12-23 20:40:40,346 - util.py line+15 - INFO - Penalty Enabled: False
# 2020-12-23 20:41:57,089 - helper.py line+48 - INFO - 10-Fold CV Average Test Success Ratio: 97.94472293727135%
# 2020-12-23 20:41:57,089 - helper.py line+49 - INFO - 10-Fold CV Average AUC Score: 0.997146672452204
# 2020-12-23 20:41:57,089 - helper.py line+50 - INFO - 10-Fold CV Average Confusion Matrix:
# [[30.5  0.6  0.   0.9]
#  [ 0.1 31.9  0.   0. ]
#  [ 0.   0.  25.8  0. ]
#  [ 0.6  0.3  0.  31.1]]
# 2020-12-23 20:41:57,091 - util.py line+39 - INFO -
# 2020-12-23 20:41:57,092 - util.py line+15 - INFO - Penalty Enabled: True
# Fitting 10 folds for each of 6 candidates, totalling 60 fits
# [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
# [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:   17.8s
# [Parallel(n_jobs=4)]: Done  60 out of  60 | elapsed:   38.9s finished
# 2020-12-23 20:42:37,129 - util.py line+32 - INFO - Best lambda value has determined as: 0.05
# 2020-12-23 20:42:46,099 - helper.py line+48 - INFO - 10-Fold CV Average Test Success Ratio: 98.2732692047148%
# 2020-12-23 20:42:46,100 - helper.py line+49 - INFO - 10-Fold CV Average AUC Score: 0.9964420688883464
# 2020-12-23 20:42:46,100 - helper.py line+50 - INFO - 10-Fold CV Average Confusion Matrix:
# [[30.8  0.5  0.   0.7]
#  [ 0.  32.   0.   0. ]
#  [ 0.   0.  25.8  0. ]
#  [ 0.6  0.3  0.  31.1]]
# 2020-12-23 20:42:46,103 - util.py line+39 - INFO -
# 2020-12-23 20:42:46,222 - app.py line+105 - INFO - Process Finished
#
# Process finished with exit code 0
