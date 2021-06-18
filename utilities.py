

from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import os
import math
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


def show_images(list_of_image_paths, ncols, plot_title=True, save=False):
    plt.box(False)
    n_imgs = len(list_of_image_paths)
    nrows = math.ceil(n_imgs / ncols)

    try:
        list_of_image_paths[ncols - 1]
    except IndexError:
        print('Error: ncols > len(images). There should be less columns than the amount of total images.')
        return

    if n_imgs == 1:
        img = Image.open(list_of_image_paths[0])
        plt.imshow(img)
        plt.axis('off')
        plt.title(list_of_image_paths[0].split('\\')[-1][:-4])

    else:

        # create figure (fig), and array of axes (ax)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

        for i, axi in enumerate(ax.flat):
            if i <= n_imgs - 1:
                img = Image.open(list_of_image_paths[i])
                title = list_of_image_paths[i].split('\\')[-1][:-4]
                axi.imshow(img, alpha=1)
                axi.axis('off')
                axi.set_title(title)

    if save:
        plt.save('plotted_imgs.png')

    plt.show()


def occurrence_plot(catalog):
    occurrences = np.zeros(42)

    for id in catalog.keys():
        images = list_pictures(os.path.join("data", id))
        images_no_au = [img for img in images if 'AU' not in img[-7:]]
        occurrences[len(images_no_au)] += 1

    sns.set_style('whitegrid')
    sns.barplot(x=list(range(42)), y=occurrences)
    plt.xlabel('Number of images')
    plt.ylabel('Number of products')
    plt.xticks(list(range(42))[::2])
    plt.show()

def number_per_category_plot(catalog):
    all_products = list(catalog.keys())
    all_cats = labels_from_ids(all_products, 'data_code/masterdata.csv')

    cats, counts = np.unique(all_cats, return_counts=True)

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(cats, counts, alpha=0.6,
            width=0.5, color=['mediumseagreen', 'green', 'lightseagreen', 'darkcyan', 'blue', 'navy'])
    plt.xlabel('Categories')
    plt.ylabel('Number of products')
    plt.xticks(list(range(len(cats))), list(cats))
    plt.show()

    '''
    #plt.style.use('seaborn')
    #sns.set_style('dark')
    sns.barplot(x=list(range(len(cats))), y=counts,saturation=.5)
    plt.xlabel('Categories')
    plt.ylabel('Number of products')
    plt.xticks(list(range(len(cats))), list(cats))
    plt.show()
    '''




if __name__ == "__main__":
    import seaborn as sns
    catalog = dict_from_json()
   # sorted, not_found = sort_by_category(catalog)
    #occurrence_plot(catalog)
    number_per_category_plot(catalog)
    #sublabels_from_ids(['590702HV-19'], 'data_code/masterdata.csv')
    pass
