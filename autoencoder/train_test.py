import os
import numpy as np

def get_train_test_split_paths(test_proportion=0.1, folder_depth=1):
    '''returns train_set, test_set'''
    filenames = []
    d = os.path.dirname(os.path.realpath(__file__))
    for _ in range(folder_depth):
        d = os.path.split(d)[0]
    d = os.path.join(d, "data")
    for root, dirs, files in os.walk(d):
        for file in files:
            if ".jpg" in file and "model_images" not in root and "not_in_master" not in root:
                filenames.append(os.path.join(root, file))

    np.random.shuffle(filenames)

    if test_proportion:
        s = int(len(filenames) // (1 / test_proportion))
    else:
        s = 0

    train_set = filenames[s:]
    test_set = filenames[:s]

    return train_set, test_set
