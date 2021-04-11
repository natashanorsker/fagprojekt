from __future__ import print_function, division
from utilities import *

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing

from torchvision.datasets import FashionMNIST

# Ignore warnings
import warnings

# set seed
random.seed(42)
seed(42)

def show_image(img_path):
    img = Image.open(img_path)
    img.show()
#show_image('../data/589338C00/589338C00_05_OG.jpg')

def get_list_IDs():
    pass


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
        image = Image.open(ID).convert('RGB')
        X = TF.to_tensor(image)
        y = self.labels[index]
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
            load_img1 = Image.open(self.data[index]).convert('RGB')
            img1 = TF.to_tensor(load_img1)
            label1 = self.labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            load_img2 = Image.open(self.data[positive_index]).convert('RGB')
            img2 = TF.to_tensor(load_img2)

            load_img3 = Image.open(self.data[negative_index]).convert('RGB')
            img3 = TF.to_tensor(load_img3)
        else:
            load_img1 = Image.open(self.data[self.test_triplets[index][0]]).convert('RGB')
            img1 = TF.to_tensor(load_img1)
            load_img2 = Image.open(self.data[self.test_triplets[index][1]]).convert('RGB')
            img2 = TF.to_tensor(load_img2)
            load_img3 = Image.open(self.data[self.test_triplets[index][2]]).convert('RGB')
            img3 = TF.to_tensor(load_img3)


        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)


def make_dataset(test_size=0.13, random_state=42):
    #first make a list of every possible image
    # get ids for the different classes [ring, earring, etc.]
    catalog = dict_from_json('../catalog.json')
    all_img_paths = []
    all_img_labels = []
    for label in catalog.keys():
        new_imgs = list_pictures(os.path.join("../data", label))
        all_img_paths += new_imgs
        all_img_labels += [label]*len(new_imgs)

    #encode the labels into integers
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(all_img_labels)

    #get partition of train and testset:
    X_train, X_test, y_train, y_test = train_test_split(all_img_paths, labels, test_size=test_size, random_state=random_state)

    X_train = X_train[:500]
    X_test = X_test[:250]
    y_train = y_train[:500]
    y_test = y_test[:250]

    #make 'generic' dataset
    training_set = Dataset(X_train, y_train)
    validation_set = Dataset(X_test, y_test)

    #make triplet dataset
    triplet_train_set = TripletDataset(training_set, train=True)
    triplet_test_set = TripletDataset(validation_set, train=False)

    return triplet_train_set, triplet_test_set, label_encoder



########
