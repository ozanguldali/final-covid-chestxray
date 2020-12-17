from sklearn.model_selection import KFold

from ml.helper import get_dataset
from ml.model import run_model

from util.garbage_util import collect_garbage
from util.logger_util import log


def main(model_name, dataset_folder, seed, cv=5, img_size=112, normalize=True):

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    log.info("Constructing datasets and arrays")
    X, y = get_dataset(dataset_folder, img_size, normalize, divide=False)

    log.info("Calling the model: " + model_name)
    run_model(model_name, X, y, kf)

    collect_garbage()


if __name__ == '__main__':
    log.info("Process Started")
    main(model_name='svm', cv=5, dataset_folder="dataset", img_size=112, normalize=True, seed=23)
    log.info("Process Finished")
