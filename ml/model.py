import sys

from ml.util import run_svm

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(model_name, X, y, seed, kf, lambdas):
    collect_garbage()

    if model_name == "svm":
        run_svm(X=X, y=y, seed=seed, kf=kf, lambdas=lambdas)

    else:
        log.fatal("ML model name is not known: " + model_name)
        sys.exit(1)
