import numpy as np
import keras
import cv2
from tensorflow.python.keras.utils.data_utils import Sequence
import os
from utilities import labels_from_ids


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_ids, batch_size=64, dim=(96, 96, 3), shuffle=True, labels=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        self.list_IDs = list_ids
        # self.n_classes = n_classes
        self.shuffle = shuffle
        self.labels = labels
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
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = cv2.imread(ID).reshape((96, 96, 3))

        X = X / 255

        if self.labels:
            l = labels_from_ids([os.path.split(ID)[-1].split("_")[0] for ID in list_IDs_temp])
            return X, l
        else:
            return X, X
