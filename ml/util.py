from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

from ml.helper import get_prediction_kf, get_best
from ml.save import save_model

from sklearn.tree import DecisionTreeClassifier

from util.logger_util import log


def run_svm(X, y, seed, penalty, kf, lambdas):
    if penalty is None:
        run_svm(X=X, y=y, seed=seed, penalty=False, kf=kf, lambdas=lambdas)
        run_svm(X=X, y=y, seed=seed, penalty=True, kf=kf, lambdas=lambdas)

    else:
        log.info("Penalty Enabled: " + str(penalty))
        tag = "Acc/SVM" + ("/LASSO" if penalty else "")

        if penalty:
            cv = kf.n_splits

            grad_dict = {
                'classifier': [LinearSVC()],
                'classifier__penalty': ['l1'],
                'classifier__C': lambdas,
                'classifier__dual': [False],
                'classifier__random_state': [seed],
                'classifier__max_iter': [1000000]
            }
            best_params = get_best(LinearSVC(), grad_dict, cv, X, y)
            best_lambda = best_params.C

            log.info("Best lambda value has determined as: " + str(best_lambda))
            svc_cv = LinearSVC(max_iter=1000000, penalty='l1', dual=False, C=best_lambda)  # probability=True

        else:
            cv = kf.n_splits

            grad_dict = {
                'classifier': [SVC()],
                'classifier__kernel': ["linear", "rbf"],  # "poly",
                'classifier__C': lambdas,
                # 'classifier__degree': [3, 4],
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

        result = get_prediction_kf(kf=kf, model=svc_cv, X=X, y=y, tag=tag)

        # save_model(result["model"], str(round(float(result["acc"]), 2)) + "_SVM_out.joblib")
        log.info("")
