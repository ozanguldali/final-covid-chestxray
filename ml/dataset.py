import os
from math import ceil

import numpy as np
from skimage.io import imread
from skimage.transform import resize

from ml import ROOT_DIR


def read_dataset(dataset_folder, resize_value=None, to_crop=False):

    dataset_dir = ROOT_DIR.split("ml")[0]

    X = []
    y = []

    def iterate_over_directory(directory, label):
        encoding = 'utf-8'

        for file in os.listdir(directory):

            try:
                file_path = str(directory + file, encoding)
            except TypeError:
                file_path = directory + file

            img_read = imread(file_path, as_gray=True)

            if to_crop:
                (y_point, x_point) = img_read.shape
                crop_size = np.min([y_point, x_point])
                start_x = x_point // 2 - (crop_size // 2)
                start_y = y_point // 2 - (crop_size // 2)

                img_read = img_read[start_y:start_y + crop_size, start_x:start_x + crop_size]

            if resize_value is not None:
                img_read = resize(img_read, resize_value, anti_aliasing=True)

            features = np.reshape(img_read, img_read.shape[0] * img_read.shape[1])

            X.append(features.tolist())
            y.append(label)

    if len(X) == 0 or len(y) == 0:
        iterate_over_directory(dataset_dir + dataset_folder + "/train/Autsim/", 0)
        iterate_over_directory(dataset_dir + dataset_folder + "/train/Normal/", 1)
        iterate_over_directory(dataset_dir + dataset_folder + "/test/Autsim/", 0)
        iterate_over_directory(dataset_dir + dataset_folder + "/test/Normal/", 1)

    return X, y


def divide_dataset(X, y):
    total_size = len(X)
    test_set_size = total_size - ceil(int(total_size * 4 / 5))

    X_tr = X[:len(X) - test_set_size]
    y_tr = y[:len(y) - test_set_size]

    X_ts = X[len(X) - test_set_size:]
    y_ts = y[len(X) - test_set_size:]

    return X_tr, y_tr, X_ts, y_ts
