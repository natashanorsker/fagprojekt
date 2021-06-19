

from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import re
import numpy as np


# https://github.com/USCDataScience/Image-Similarity-Deep-Ranking/blob/master/triplet_sampler.py
def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    '''
    Returns path of folder/directory of every image present in directory by using simple regex expression
    '''
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def dict_from_json(path="catalog.json"):
    # open the product catalog:
    with open(path, "r") as a_file:
        catalog = json.loads(a_file.read())
    return catalog


def trim(im):
    # from https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    # Bounding box given as a 4-tuple defining the left, upper, right, and lower pixel coordinates.
    # If the image is completely empty, this method returns None.
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def labels_from_ids(ids, master_file_path='../data_code/masterdata.csv', label_encoder=None):
    # should return the category in a string
    df = pd.read_csv(master_file_path, sep=';')
    df.columns = df.columns.str.lower()

    if label_encoder:
        ids = label_encoder.inverse_transform(ids)

    labels = []
    for id in ids:
        try:
            label = df['item category'].loc[df.key == id].values[0]

            if label not in ['Bracelets', 'Charms', 'Jewellery spare parts', 'Necklaces & Pendants', 'Rings', 'Earrings']:
                label = 'Misc'

        except:
            print("Can't find info on product: {}".format(id))
            label = 'Set'
        labels.append(label)

    return labels

def sublabels_from_ids(ids, master_file_path='../data_code/masterdata.csv', label_encoder=None):
    # should return the category in a string
    df = pd.read_csv(master_file_path, sep=';')
    df.columns = df.columns.str.lower()

    if label_encoder:
        ids = label_encoder.inverse_transform(ids)

    labels = []
    for id in ids:
        try:
            label = df['item sub-category'].loc[df.key == id].values[0].lower()

        except:
            print("Can't find info on product: {}".format(id))
            label = 'Set'
        labels.append(label)
    return labels



def sublabels_from_ids(ids, master_file_path='../data_code/masterdata.csv', label_encoder=None):
    # should return the category in a string
    df = pd.read_csv(master_file_path, sep=';')
    df.columns = df.columns.str.lower()

    if label_encoder:
        ids = label_encoder.inverse_transform(ids)

    labels = []
    for id in ids:
        try:
            label = df['item sub-category'].loc[df.key == id].values[0].lower()

        except:
            print("Can't find info on product: {}".format(id))
            label = 'Set'
        labels.append(label)
    return labels


def sort_by_category(catalog, category='item category', master_file_path='data_code/masterdata.csv', save=True):
    # read the master file:


    df = pd.read_csv(master_file_path, sep=';')
    df.columns = df.columns.str.lower()
    category = category.lower()

    keys = df[category].unique()
    categories = {k: [] for k in keys}
    not_found_products = []

    for product in catalog.keys():
        cat = labels_from_ids(product, name_of_info=category, master_file_path=master_file_path)
        if cat is None:
            not_found_products += [product]
            pass

        else:
            categories[cat] += [product]

    # remove empty categories:
    sorted = {key: value for key, value in categories.items() if len(value) != 0}

    if save:
        a_file = open("catalog_by_category.json", "w")
        json.dump(sorted, a_file)
        a_file.close()

        another_file = open("id_not_in_masterfile.json", "w")
        json.dump(not_found_products, another_file)
        another_file.close()

    return sorted, not_found_products
