from joblib import dump

from ml import ROOT_DIR


def save_model(model, filename):
    dump(model, ROOT_DIR + "/saved_models/" + filename)
