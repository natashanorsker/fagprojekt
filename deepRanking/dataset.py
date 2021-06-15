from __future__ import print_function, division
from autoencoder.train_test import get_train_test_split_paths
from utilities import list_pictures, labels_from_ids
import json
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import random
import numpy as np
from PIL import Image
from numpy.random import seed

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

# Ignore warnings
import warnings

# set seed
random.seed(42069)
seed(42069)


def list_paths_labels(folder_depth=1):
    # first make a list of every possible image
    # get ids for the different classes [ring, earring, etc.]
    # catalog = dict_from_json('../catalog.json')
    all_img_paths = []
    all_img_labels = []

    root = os.path.dirname(os.path.realpath(__file__))
    for _ in range(folder_depth):
        root = os.path.split(root)[0]

    catalog = json.loads(open(os.path.join(root, "catalog.json"), "r").read())

    root = os.path.join(root, "data")

    for label in catalog.keys():
        new_imgs = list_pictures(os.path.join(root, label))
        all_img_paths += new_imgs
        all_img_labels += [label] * len(new_imgs)

    return all_img_paths, all_img_labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, label_encoder):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.label_encoder = label_encoder

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def add_class_labels(self):
        raise NotImplementedError(':(')

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        image = Image.open(ID)
        X = TF.to_tensor(image)
        y = self.labels[index]
        if X.size()[0] != 3:
            image = Image.open(ID).convert('RGB')
            X = TF.to_tensor(image)
        return X, y


def make_dataset(label_encoder, n_val_products, NoDuplicates=False, folder_depth=1):
    '''NoDuplicates selects every 40th elements in the dataset,
     so no augmented images are retrieved
     
     return: train_set, test_set'''

    # make sure that our test set is not used for training and validation:
    test_set_paths = get_train_test_split_paths(folder_depth=1)[1]

    # get all possible images:
    all_img_paths, all_img_labels = list_paths_labels(folder_depth=folder_depth)

    # remove test images from pool:
    # indices to remove:
    indices = [i for i, e in enumerate(all_img_paths) if e in test_set_paths]
    all_img_paths = [i for j, i in enumerate(all_img_paths) if j not in indices]
    all_img_labels = [i for j, i in enumerate(all_img_labels) if j not in indices]

    labels = label_encoder.transform(all_img_labels)
    labels_set = list(set(labels))
    label_to_indices = {label: np.where(labels == label)[0] for label in labels_set}

    test_products = np.random.choice(labels_set, n_val_products, replace=False)
    train_products = set(labels_set) - set(test_products)

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for test_product in test_products:
        indices = label_to_indices[test_product]
        X_val += [all_img_paths[idx] for idx in indices]
        y_val += [labels[idx] for idx in indices]

    for train_product in train_products:
        indices = label_to_indices[train_product]
        X_train += [all_img_paths[idx] for idx in indices]
        y_train += [labels[idx] for idx in indices]

    if NoDuplicates:
        y_val = np.array(y_val)[::40]
        y_train = np.array(y_train)[::40]
        X_val = np.array(X_val)[::40]
        X_train = np.array(X_train)[::40]

    else:
        y_val = np.array(y_val)
        y_train = np.array(y_train)

    training_set = Dataset(X_train, y_train, label_encoder)
    validation_set = Dataset(X_val, y_val, label_encoder)

    return training_set, validation_set


##########

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        self.dataset = dataset
        self.labels = self.dataset.labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
