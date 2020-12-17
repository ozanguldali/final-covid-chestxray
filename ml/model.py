import sys

from ml.util import run_svm, run_lr, run_knn

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(model_name, X, y, seed, kf, lambdas):
    collect_garbage()

    if model_name == "svm":
        run_svm(X=X, y=y, seed=seed, kf=kf, lambdas=lambdas)

    elif model_name == "lr":
        run_lr(X=X, y=y, kf=kf)

    elif model_name == "knn":
        run_knn(X=X, y=y, kf=kf)

    elif model_name == "all":
        log.info("Running ML model: svm")
        run_svm(X=X, y=y, kf=kf)

        log.info("Running ML model: lr")
        run_lr(X=X, y=y, kf=kf)

        log.info("Running ML model: knn")
        run_knn(X=X, y=y, kf=kf)

    else:
        log.fatal("ML model name is not known: " + model_name)
        sys.exit(1)
