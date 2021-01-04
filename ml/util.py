from sklearn.svm import SVC

from ml.helper import get_prediction_kf, get_best
from util.logger_util import log


def run_svm(X, y, seed, kf, lambdas):

    tag = "Acc/SVM"
    cv = kf.n_splits

    grad_dict = {
        'classifier': [SVC()],
        'classifier__kernel': ["linear", "rbf"],
        'classifier__C': lambdas,
        'classifier__gamma': ['scale', 'auto'],
        'classifier__decision_function_shape': ['ovo', 'ovr'],
        'classifier__random_state': [seed]
    }
    best_params = get_best(SVC(), grad_dict, cv, X, y)
    C, kernel, gamma, decision_function_shape = best_params.C, best_params.kernel, best_params.gamma, best_params.decision_function_shape
    log.info("best params:\nC: {}\tkernel: {}\tgamma: {}\tdecision_function_shape: {}".format(C, kernel, gamma,
                                                                                              decision_function_shape))
    svc_cv = SVC(probability=True, C=C, kernel=kernel, gamma=gamma,
                 decision_function_shape=decision_function_shape)

    # svc_cv = SVC(max_iter=1000000, probability=True)

    get_prediction_kf(kf=kf, model=svc_cv, X=X, y=y, tag=tag)

    # save_model(result["model"], str(round(float(result["acc"]), 2)) + "_SVM_out.joblib")
    log.info("")
