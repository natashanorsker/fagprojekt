from __future__ import print_function, division
from utilities import list_pictures
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
random.seed(420)
seed(420)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

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


class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset, train=True):
        self.dataset = dataset
        self.train = train

        self.labels = self.dataset.labels
        self.data = self.dataset.list_IDs
        # generate fixed triplets
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.data))]
            self.test_triplets = triplets

    def __getitem__(self, index):

        if self.train:
            load_img1 = Image.open(self.data[index])
            img1 = TF.to_tensor(load_img1)
            anchor_label = self.labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[anchor_label])
            negative_label = np.random.choice(list(self.labels_set - set([anchor_label])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            load_img2 = Image.open(self.data[positive_index])
            img2 = TF.to_tensor(load_img2)

            load_img3 = Image.open(self.data[negative_index])
            img3 = TF.to_tensor(load_img3)

            if not all(len(img) == 3 for img in [img1, img2, img3]):
                load_img1 = Image.open(self.data[index]).convert('RGB')
                img1 = TF.to_tensor(load_img1)

                load_img2 = Image.open(self.data[positive_index]).convert('RGB')
                img2 = TF.to_tensor(load_img2)

                load_img3 = Image.open(self.data[negative_index]).convert('RGB')
                img3 = TF.to_tensor(load_img3)

        else:
            load_img1 = Image.open(self.data[self.test_triplets[index][0]])
            img1 = TF.to_tensor(load_img1)
            load_img2 = Image.open(self.data[self.test_triplets[index][1]])
            img2 = TF.to_tensor(load_img2)
            load_img3 = Image.open(self.data[self.test_triplets[index][2]])
            img3 = TF.to_tensor(load_img3)

            if not all(len(img) == 3 for img in [img1, img2, img3]):
                load_img1 = Image.open(self.data[self.test_triplets[index][0]]).convert('RGB')
                img1 = TF.to_tensor(load_img1)
                load_img2 = Image.open(self.data[self.test_triplets[index][1]]).convert('RGB')
                img2 = TF.to_tensor(load_img2)
                load_img3 = Image.open(self.data[self.test_triplets[index][2]]).convert('RGB')
                img3 = TF.to_tensor(load_img3)

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)


def list_paths_labels():
    # first make a list of every possible image
    # get ids for the different classes [ring, earring, etc.]
    catalog = json.loads(open('../catalog.json', "r").read())
    # catalog = dict_from_json('../catalog.json')
    all_img_paths = []
    all_img_labels = []
    for label in catalog.keys():
        new_imgs = list_pictures(os.path.join("../data", label))
        all_img_paths += new_imgs
        all_img_labels += [label] * len(new_imgs)

    return all_img_paths, all_img_labels

"""
def make_dataset(label_encoder, test_size=0.13, random_state=42):
    all_img_paths, all_img_labels = list_paths_labels()
    # encode the labels into integers
    labels = label_encoder.transform(all_img_labels)

    # get partition of train and testset:
    X_train, X_test, y_train, y_test = train_test_split(all_img_paths, labels, test_size=test_size,
                                                        random_state=random_state)

    # make 'generic' dataset
    training_set = Dataset(X_train, y_train)
    validation_set = Dataset(X_test, y_test)

    return training_set, validation_set
"""

def make_dataset(label_encoder, n_test_products):
    all_img_paths, all_img_labels = list_paths_labels()
    labels = label_encoder.transform(all_img_labels)

    labels_set = list(set(labels))
    label_to_indices = {label: np.where(labels == label)[0] for label in labels_set}

    classes = np.random.choice(labels_set, n_test_products, replace=False)

    test_products = np.random.choice(labels_set, n_test_products, replace=False)
    train_products = set(labels_set) - set(test_products)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for test_product in test_products:
        indices = label_to_indices[test_product]
        X_test += [all_img_paths[idx] for idx in indices]
        y_test += [labels[idx] for idx in indices]

    for train_product in train_products:
        indices = label_to_indices[train_product]
        X_train += [all_img_paths[idx] for idx in indices]
        y_train += [labels[idx] for idx in indices]

    y_test = np.array(y_test)
    y_train = np.array(y_train)

    #maybe delete later idk:
    """
    X_train = []
    y_train = []
    for class_ in classes:
        indices = label_to_indices[class_]
        X_train += [all_img_paths[idx] for idx in indices]
        y_train += [labels[idx] for idx in indices]
    y_train = np.array(y_train)
    plot_dataset = Dataset(X_train, y_train)
    ##
    """

    training_set = Dataset(X_train, y_train)
    validation_set = Dataset(X_test, y_test)

    return training_set, validation_set


########

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
