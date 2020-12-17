from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from ml.helper import get_prediction_kf, get_best_lambda

from util.logger_util import log


def run_svm(X, y, seed, kf, lambdas):
    cv = kf.n_splits
    tag = "Acc/SVM"

    grad_dict = {
        'classifier': [LinearSVC()],
        'classifier__penalty': ['l1'],
        'classifier__C': lambdas,
        'classifier__dual': [False],
        'classifier__random_state': [seed],
        'classifier__max_iter': [10000]
    }
    bests = get_best_lambda(LinearSVC(), grad_dict, cv, X, y)
    best_lambda = lambdas[bests.best_index_]

    log.info("Best lambda value has determined as: " + str(best_lambda))
    svc_cv = LinearSVC(max_iter=100000, penalty='l1', dual=False, C=best_lambda)  # probability=True

    get_prediction_kf(kf=kf, model=svc_cv, X=X, y=y, tag=tag)
    log.info("")


def run_lr(X, y, seed, kf, lambdas):
    cv = kf.n_splits
    tag = "Acc/LR"

    grad_dict = {
        'classifier': [LogisticRegression()],
        'classifier__penalty': ['l1'],
        'classifier__C': lambdas,
        'classifier__solver': ["liblinear"],
        'classifier__random_state': [seed],
        'classifier__max_iter': [10000]
    }
    bests = get_best_lambda(LogisticRegression(), grad_dict, cv, X, y)
    best_lambda = lambdas[bests.best_index_]
    log.info("Best lambda value has determined as: " + str(best_lambda))
    clf_cv = LogisticRegression(max_iter=100000, solver='liblinear', penalty='l1', C=best_lambda)

    get_prediction_kf(kf, clf_cv, X, y, tag)
    log.info("")


def run_knn(X, y, seed, kf, lambdas):
    tag = "Acc/KNN"

    neigh_cv = KNeighborsClassifier(n_neighbors=len(set(y)))

    get_prediction_kf(kf, neigh_cv, X, y, tag)
    log.info("")
