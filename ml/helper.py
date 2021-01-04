import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm

from ml.dataset import read_dataset, divide_dataset

from util.logger_util import log
from util.tensorboard_util import writer


def get_prediction_kf(kf, model, X, y, tag=None):
    cv = kf.n_splits
    ratios = []
    conf_matrices = []
    roc_list = []
    for e, (train, test) in enumerate(tqdm(kf.split(X, y))):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        # log.info("iteration - {}".format(e))
        X_train, X_test, y_train, y_test = X[train], X[test], np.array(y)[train], np.array(y)[test]
        # log.info("prepared data")
        model.fit(X_train, y_train)
        # log.info("model has been fitted")
        success_ratio = model.score(X_test, y_test)
        # log.info("got the score")
        # log.info(str(cv) + "-Fold CV -- Iteration " + str(e) + " Test Success Ratio: " + str(100*success_ratio) + "%")
        ratios.append(success_ratio)

        test_prob = model._predict_proba_lr(X_test) if isinstance(model, LinearSVC) else model.predict_proba(X_test)
        auc = roc_auc_score(y_test, test_prob, multi_class="ovr")
        # log.info(str(cv) + "-Fold CV -- Iteration " + str(e) + " AUC Score: " + str(auc))
        roc_list.append(auc)

        conf_matrix = confusion_matrix(y_test.tolist(), model.predict(X_test).tolist())
        # log.info(str(cv) + "-Fold CV -- Iteration " + str(e) + " Confusion Matrix:\n" + str(conf_matrix))
        conf_matrices.append(conf_matrix)

        if tag is not None:
            writer.add_scalar(tag, success_ratio, e)

    log.info(str(cv) + "-Fold CV Average Test Success Ratio: " + str(100 * np.average(np.array(ratios))) + "%")
    log.info(str(cv) + "-Fold CV Average AUC Score: " + str(np.average(np.array(roc_list))))
    log.info(str(cv) + "-Fold CV Average Confusion Matrix:\n" + str(np.mean(conf_matrices, axis=0)))

    return {"model": model, "acc": str(100 * np.average(np.array(ratios)))}


def get_dataset(dataset_folder, img_size, normalize, divide=False):
    log.info("Reading dataset")
    X, y = read_dataset(dataset_folder=dataset_folder, resize_value=(img_size, img_size), to_crop=True)

    if normalize:
        X = StandardScaler().fit_transform(X)

    if divide:
        log.info("Dividing dataset into train and test data")
        X_tr, y_tr, X_ts, y_ts = divide_dataset(X, y)
        log.info("Train data length: %d" % len(y_tr))
        log.info("Test data length: %d" % len(y_ts))

        return X_tr, y_tr, X_ts, y_ts

    return X, y


def get_best(classifier, grad_dict, cv, X, y):
    pipe = Pipeline([('classifier', classifier)])

    param_grid = [grad_dict]

    clf = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=1, n_jobs=4, pre_dispatch=2*4)

    fit = clf.fit(X, y)

    return fit.best_params_['classifier']

