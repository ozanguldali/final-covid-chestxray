from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

from ml.helper import get_prediction_kf, get_best_lambda
from ml.save import save_model

from sklearn.tree import DecisionTreeClassifier

from util.logger_util import log


def run_svm(X, y, seed, penalty, kf=None, lambdas=None):
    if penalty is None:
        run_svm(X=X, y=y, seed=seed, penalty=False, kf=kf)
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
            bests = get_best_lambda(LinearSVC(), grad_dict, cv, X, y)
            best_lambda = lambdas[bests.best_index_]

            log.info("Best lambda value has determined as: " + str(best_lambda))
            svc_cv = LinearSVC(max_iter=1000000, penalty='l1', dual=False, C=best_lambda)  # probability=True

        else:
            svc_cv = SVC(max_iter=1000000, probability=True)

        result = get_prediction_kf(kf=kf, model=svc_cv, X=X, y=y, tag=tag)

        # save_model(result["model"], str(round(float(result["acc"]), 2)) + "_SVM_out.joblib")
        log.info("")


def run_lr(X, y, seed, kf, penalty, lambdas):
    if penalty is None:
        run_svm(X=X, y=y, seed=seed, penalty=False, kf=kf)
        run_svm(X=X, y=y, seed=seed, penalty=True, kf=kf, lambdas=lambdas)

    else:
        log.info("Penalty Enabled: " + str(penalty))
        tag = "Acc/SVM" + ("/LASSO" if penalty else "")

        if penalty:
            cv = kf.n_splits

            grad_dict = {
                'classifier': [LogisticRegression()],
                'classifier__penalty': ['l1'],
                'classifier__C': lambdas,
                'classifier__solver': ["liblinear"],
                'classifier__random_state': [seed],
                'classifier__max_iter': [1000000]
            }
            bests = get_best_lambda(LogisticRegression(), grad_dict, cv, X, y)
            best_lambda = lambdas[bests.best_index_]
            log.info("Best lambda value has determined as: " + str(best_lambda))
            clf_cv = LogisticRegression(max_iter=1000000, solver='liblinear', penalty='l1', C=best_lambda)

        else:
            clf_cv = LogisticRegression(max_iter=1000000, solver='liblinear')

        get_prediction_kf(kf, clf_cv, X, y, tag)
        log.info("")


def run_dt(X, y, kf):
    dt_clf = DecisionTreeClassifier()
    get_prediction_kf(kf, dt_clf, X, y, "tag")
    log.info("")
