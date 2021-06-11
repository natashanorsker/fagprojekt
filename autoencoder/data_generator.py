import numpy as np
import keras
import cv2
from tensorflow.python.keras.utils.data_utils import Sequence
import os
from utilities import labels_from_ids

this_file_path = os.path.dirname(os.path.realpath(__file__))

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_ids, batch_size=64, dim=(96, 96, 3), shuffle=True, labels=False, ids=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        self.list_IDs = list_ids
        # self.n_classes = n_classes
        self.shuffle = shuffle
        self.labels = labels
        self.ids = ids
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = cv2.imread(ID).reshape((96, 96, 3))

        X = X / 256

        if self.labels:
            l = labels_from_ids([os.path.split(ID)[-1].split("_")[0] for ID in list_IDs_temp])
            return X, l
        elif self.ids:
            return X, list_IDs_temp
        else:
            return X, X


def get_train_test_split_paths(test_proportion=0.1):
    filenames = []
    d = os.path.join(os.path.split(this_file_path)[0], "data")
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


