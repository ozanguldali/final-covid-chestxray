import csv
import os
from math import ceil, floor
from random import randrange, shuffle
from shutil import copyfile, rmtree

import numpy as np

ROOT_DIR = str(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = "/Users/ozanguldali/Documents/master_courses/deep_learning/final_project/archive/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset"
METADATA_PATH = ROOT_DIR + "/Chest_xray_Corona_Metadata.csv"

train_viral_covid19_folder = ROOT_DIR+'/dataset/train/Viral-COVID19/'
train_viral_other_folder = ROOT_DIR+'/dataset/train/Viral-Other/'
train_bacterial_folder = ROOT_DIR+'/dataset/train/Bacterial/'
train_normal_folder = ROOT_DIR+'/dataset/train/Normal/'
test_viral_covid19_folder = ROOT_DIR+'/dataset/test/Viral-COVID19/'
test_viral_other_folder = ROOT_DIR+'/dataset/test/Viral-Other/'
test_bacterial_folder = ROOT_DIR+'/dataset/test/Bacterial/'
test_normal_folder = ROOT_DIR+'/dataset/test/Normal/'

# whole dataset_unique list init
covid_chestxray_dataset = []


def scrape_metadata():
    viral_covid_patients = {}
    viral_other_patients = {}
    bacterial_patients = {}
    normal_patients = {}

    # read metadata file
    with open(METADATA_PATH, mode='r') as csv_data:
        reader = csv.DictReader(csv_data, delimiter=',')

        for row in reader:
            covid_chestxray_dataset.append(row)

    # filter dataset_unique for COVID-19 patients having sex and age info, and PA X-ray image
    for data in covid_chestxray_dataset:
        img = data["X_ray_image_name"]
        dataset_type = data["Dataset_type"]
        if "Normal" == str(data["Label"]):
            normal_patients[img] = dataset_type
        else:
            if "bacteria" == str(data["Label_1_Virus_category"]):
                if "" == str(data["Label_2_Virus_category"]):
                    bacterial_patients[img] = dataset_type
                else:
                    continue
            elif "Virus" == str(data["Label_1_Virus_category"]):
                if "COVID-19" == str(data["Label_2_Virus_category"]):
                    viral_covid_patients[img] = dataset_type
                elif "" == str(data["Label_2_Virus_category"]):
                    viral_other_patients[img] = dataset_type
                else:
                    continue
            else:
                continue

    return viral_covid_patients, viral_other_patients, bacterial_patients, normal_patients


def dataset_investigate():
    lists = scrape_metadata()

    print("not unique viral-covid: ", len(lists[0]))
    print("not unique viral-other: ", len(lists[1]))
    print("not unique bacterial: ", len(lists[2]))
    print("not unique normal: ", len(lists[3]))


def construct_dataset(default_size=240, reset=False, create=False):
    if reset:
        prepare_directory(train_viral_covid19_folder)
        prepare_directory(train_viral_other_folder)
        prepare_directory(train_bacterial_folder)
        prepare_directory(train_normal_folder)
        prepare_directory(test_viral_covid19_folder)
        prepare_directory(test_viral_other_folder)
        prepare_directory(test_bacterial_folder)
        prepare_directory(test_normal_folder)

    lists = scrape_metadata()

    viral_covid_dict = lists[0]
    viral_covid_images = list(viral_covid_dict.keys())
    viral_covid_types = list(viral_covid_dict.values())
    len_viral_covid = len(viral_covid_dict)
    viral_other_dict = lists[1]
    viral_other_images = list(viral_other_dict.keys())
    viral_other_types = list(viral_other_dict.values())
    len_viral_other = len(viral_other_dict)
    bacterial_dict = lists[2]
    bacterial_images = list(bacterial_dict.keys())
    bacterial_types = list(bacterial_dict.values())
    len_bacterial = len(bacterial_dict)
    normal_dict = lists[3]
    normal_images = list(normal_dict.keys())
    normal_types = list(normal_dict.values())
    len_normal = len(normal_dict)

# ---------------------------------------------------------------------------------------------------------------------

    len_train_viral_covid = 40
    len_test_viral_covid = 18

    len_train_viral_other = len_train_bacterial = len_train_normal = default_size
    len_test_viral_other = len_test_bacterial = len_test_normal = int(default_size / 3)

    # construct train and test sets
    train_viral_covid, train_viral_other, train_bacterial, train_normal = {}, {}, {}, {}
    train_size = len_train_viral_covid + len_train_viral_other + len_train_bacterial + len_train_normal

    test_viral_covid, test_viral_other, test_bacterial, test_normal = {}, {}, {}, {}
    test_size = len_test_viral_covid + len_test_viral_other + len_test_bacterial + len_test_normal

    viral_covid_iter = viral_other_iter = bacterial_iter = normal_iter = 0
    for i in range(train_size):
        if viral_covid_iter < len_train_viral_covid:
            train_viral_covid[viral_covid_images[viral_covid_iter]] = viral_covid_types[viral_covid_iter]
            viral_covid_iter += 1
        elif viral_other_iter < len_train_viral_other:
            train_viral_other[viral_other_images[viral_other_iter]] = viral_other_types[viral_other_iter]
            viral_other_iter += 1
        elif bacterial_iter < len_train_bacterial:
            train_bacterial[bacterial_images[bacterial_iter]] = bacterial_types[bacterial_iter]
            bacterial_iter += 1
        else:
            train_normal[normal_images[normal_iter]] = normal_types[normal_iter]
            normal_iter += 1

    for i in range(test_size):
        if viral_covid_iter < len_train_viral_covid + len_test_viral_covid:
            test_viral_covid[viral_covid_images[viral_covid_iter]] = viral_covid_types[viral_covid_iter]
            viral_covid_iter += 1
        elif viral_other_iter < len_train_viral_other + len_test_viral_other:
            test_viral_other[viral_other_images[viral_other_iter]] = viral_other_types[viral_other_iter]
            viral_other_iter += 1
        elif bacterial_iter < len_train_bacterial + len_test_bacterial:
            test_bacterial[bacterial_images[bacterial_iter]] = bacterial_types[bacterial_iter]
            bacterial_iter += 1
        else:
            test_normal[normal_images[normal_iter]] = normal_types[normal_iter]
            normal_iter += 1

    print(len(train_viral_covid))
    print(len(train_viral_other))
    print(len(train_bacterial))
    print(len(train_normal))

    print(len(test_viral_covid))
    print(len(test_viral_other))
    print(len(test_bacterial))
    print(len(test_normal))

    if create:
        construct_related_base_directory(train_viral_covid, train_viral_covid19_folder)
        construct_related_base_directory(train_viral_other, train_viral_other_folder)
        construct_related_base_directory(train_bacterial, train_bacterial_folder)
        construct_related_base_directory(train_normal, train_normal_folder)

        construct_related_base_directory(test_viral_covid, test_viral_covid19_folder)
        construct_related_base_directory(test_viral_other, test_viral_other_folder)
        construct_related_base_directory(test_bacterial, test_bacterial_folder)
        construct_related_base_directory(test_normal, test_normal_folder)


def split_main_dataset(reset=False, create=False):
    viral_covid19_folder = SOURCE_DIR + "/viral_covid/"
    viral_other_folder = SOURCE_DIR + "/viral_other/"
    bacterial_folder = SOURCE_DIR + "/bacterial/"
    normal_folder = SOURCE_DIR + "/normal/"

    if reset:
        prepare_directory(viral_covid19_folder)
        prepare_directory(viral_other_folder)
        prepare_directory(bacterial_folder)
        prepare_directory(normal_folder)

    lists = scrape_metadata()

    viral_covid_dict = lists[0]
    viral_other_dict = lists[1]
    bacterial_dict = lists[2]
    normal_dict = lists[3]

    if create:
        construct_related_base_directory(viral_covid_dict, viral_covid19_folder)
        construct_related_base_directory(viral_other_dict, viral_other_folder)
        construct_related_base_directory(bacterial_dict, bacterial_folder)
        construct_related_base_directory(normal_dict, normal_folder)


def elect_from_larger_dataset(small, large, dataset):
    elected = []
    rand = []
    rand_range = len(small)

    for _ in range(rand_range):
        r = randrange(rand_range)
        while r in rand:
            r = randrange(rand_range)
        rand.append(r)
        elected.append(large[r])

    refuse = list(set(large) - set(elected))

    dataset = remove_refuse_info_list_from_list(refuse, "id", dataset)

    return elected, dataset


def remove_refuse_info_list_from_list(refuse, key, target):
    temp_list = []
    for data in target:
        if data[key] not in refuse:
            temp_list.append(data)

    target.clear()
    target.extend(temp_list)
    temp_list.clear()

    return target


def prepare_directory(folder):
    if os.path.exists(folder):
        if len(os.listdir(folder)) != 0:
            clear_directory(folder)
    else:
        create_directory(folder)


def create_directory(folder):
    os.makedirs(folder)


def clear_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def construct_related_base_directory(data_dict, folder):
    for img in data_dict:
        source = SOURCE_DIR + "/" + data_dict[img] + "/" + img
        destination = folder + img
        copyfile(source, destination)


if __name__ == '__main__':
    # RUN JUST ONE TIME
    # construct_dataset(reset=False, create=False)
    dataset_investigate()
