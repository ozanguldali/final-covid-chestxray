import sys

from ml.util import run_svm, run_lr, run_dt

from util.garbage_util import collect_garbage
from util.logger_util import log


def run_model(model_name, X, y, seed, kf, penalty, lambdas):
    collect_garbage()
    lambdas = (lambdas if (penalty or penalty is None) else None)

    if model_name == "svm":
        run_svm(X=X, y=y, seed=seed, penalty=penalty, kf=kf, lambdas=lambdas)

    elif model_name == "lr":
        run_lr(X=X, y=y, kf=kf, seed=seed, penalty=penalty, lambdas=lambdas)

    elif model_name == "dt":
        run_dt(X=X, y=y, kf=kf)

    elif model_name == "all":
        log.info("Running ML model: svm")
        run_svm(X=X, y=y, seed=seed, penalty=penalty, kf=kf, lambdas=lambdas)

        log.info("Running ML model: lr")
        run_lr(X=X, y=y, kf=kf, seed=seed, penalty=penalty, lambdas=lambdas)

    else:
        log.fatal("ML model name is not known: " + model_name)
        sys.exit(1)
