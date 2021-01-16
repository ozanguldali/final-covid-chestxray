from sklearn.model_selection import KFold

from ml.helper import get_dataset
from ml.model import run_model

from util.garbage_util import collect_garbage
from util.logger_util import log


def main(seed, model_name="svm", dataset_folder="dataset", cv=10, img_size=224, normalize=True, lambdas=None):

    if lambdas is None:
        lambdas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    log.info("Constructing datasets and arrays")
    X, y = get_dataset(dataset_folder, img_size, normalize, divide=False)

    log.info("Calling the model: " + model_name)
    run_model(model_name=model_name, X=X, y=y, seed=seed, kf=kf, lambdas=lambdas)

    collect_garbage()


if __name__ == '__main__':
    log.info("Process Started")
    main(model_name='dt', cv=10, dataset_folder="dataset", seed=17)
    log.info("Process Finished")

# SVM
# 2021-01-01 19:34:50,792 - helper.py line+51 - INFO - 10-Fold CV Average Test Success Ratio: 84.64909903807072%
# 2021-01-01 19:34:50,795 - helper.py line+52 - INFO - 10-Fold CV Average AUC Score: 0.9678050463943955
# 2021-01-01 19:34:50,798 - helper.py line+53 - INFO - 10-Fold CV Average Confusion Matrix:
# [[25.5  1.4  0.5  4.6]
#  [ 0.3 30.   0.2  1.5]
#  [ 0.2  0.1 24.9  0.6]
#  [ 6.8  1.8  0.7 22.7]]
