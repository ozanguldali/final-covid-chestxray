import os
import sys

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


def main(transfer_learning, method="", ml_model_name="", cv=10, dataset_folder="dataset", penalty=None,
         pretrain_file=None, batch_size=8, img_size=112, num_workers=4, cnn_model_name="", optimizer_name='Adam',
         validation_freq=0.1, lr=0.001, momentum=0.9, partial=0.125, betas=(0.9, 0.99), weight_decay=0.025,
         update_lr=True, is_pre_trained=False, fine_tune=False, num_epochs=16, normalize=True, lambdas=None, seed=1):

    if lambdas is None:
        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    if not transfer_learning:
        if method.lower() == "ml":
            run_ML.main(model_name=ml_model_name, dataset_folder=dataset_folder, seed=seed, cv=cv,
                        img_size=img_size, normalize=normalize)
        elif method.lower() == "cnn":
            run_CNN.main(save=False, dataset_folder=dataset_folder, batch_size=batch_size, test_without_train=False,
                         img_size=img_size, num_workers=num_workers, num_epochs=num_epochs, model_name=cnn_model_name,
                         optimizer_name=optimizer_name, is_pre_trained=is_pre_trained, fine_tune=fine_tune,
                         update_lr=update_lr, normalize=normalize, validation_freq=validation_freq, lr=lr,
                         momentum=momentum, partial=partial, betas=betas, weight_decay=weight_decay)
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

        if is_pre_trained and pretrain_file is not None and \
                cnn_model_name in pretrain_file.lower():
            log.info("Getting PreTrained CNN model: " + cnn_model_name + " from the Weights of " + pretrain_file)
            model = cnn_model.weighted_model(cnn_model_name, pretrain_file)

        else:
            log.info("Running CNN model: " + cnn_model_name)
            model = cnn_model.run_model(model_name=cnn_model_name, optimizer_name=optimizer_name, fine_tune=fine_tune,
                                        is_pre_trained=is_pre_trained, train_loader=train_loader, num_epochs=num_epochs,
                                        test_loader=test_loader, validation_freq=validation_freq, lr=lr,
                                        momentum=momentum, partial=partial, betas=betas, weight_decay=weight_decay,
                                        update_lr=update_lr, save=False, dataset_folder=dataset_folder)

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
        log.info("Total class 0 size: " + str(class0_size))
        log.info("Total class 1 size: " + str(class1_size))

        if normalize:
            X_cnn = Normalizer().fit_transform(X_cnn)
        X_cnn = StandardScaler().fit_transform(X_cnn)

        log.info("Number of features in X_cnn: " + str(len(X_cnn[0])))

        kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

        ml_model.run_model(model_name=ml_model_name, X=X_cnn, y=y, seed=seed, kf=kf, penalty=penalty, lambdas=lambdas)

    collect_garbage()


if __name__ == '__main__':
    log.info("Process Started")
    main(transfer_learning=False, method="ml", ml_model_name="all", cnn_model_name="resnet18", is_pre_trained=True,
         dataset_folder="dataset", pretrain_file="84.35_PreTrained_resnet18_Adam_dataset_out", img_size=112,
         cv=10, seed=17)

    log.info("Process Finished")
