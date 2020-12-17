import os
from shutil import rmtree


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


def path_exists(folder, file, check_type="equals"):
    check = lambda s1, s2: (s1 == s2) if check_type == "equals" else (s1 in s2)

    exist_files = []
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if len(files) > 0:
            for f in files:
                if check(file, f):
                    exist_files.append(f)
        else:
            assert("no files in directory: " + folder)
    else:
        assert("folder does NOT exist: " + folder)

    return exist_files
