from utilities import dict_from_json, labels_from_ids, list_pictures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import math
from PIL import Image, ImageChops
import os
import cv2
from scipy.linalg import svd
import json



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


def number_per_category_plot(catalog):
    all_products = list(catalog.keys())
    all_cats = labels_from_ids(all_products, '../data_code/masterdata.csv')

    cats, counts = np.unique(all_cats, return_counts=True)
    cats[4] = 'Necklaces \n Pendants'
    # creating the bar plot
    plt.style.use('seaborn')
    plt.figure(figsize=(9, 4))

    plt.bar(cats, counts, alpha=0.6,
            width=0.5,linewidth =0.5,  color=['green', 'mediumseagreen', 'lightseagreen', 'darkcyan', 'cornflowerblue', 'navy'])
    plt.xlabel('Categories', fontsize=16)
    plt.ylabel('Number of products', fontsize=16)
    plt.xticks(list(range(len(cats))), list(cats), fontsize=14)
    #plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.9)
    plt.savefig('num_per_cat.png');
    plt.tight_layout()
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


def PCAplot(catalog):
    include = set(['bracelets', 'charms', 'necklaces', 'rings', 'earrings'])
    data = np.zeros((len(catalog), 96 * 96 * 3), dtype='uint8')

    labels = [None] * len(catalog)
    for i, folder in enumerate(catalog.keys()):
        if folder in os.listdir('../data'):
            label = catalog[folder]['product_category']
            if label in include:
                file_ = '../data' + '/' + folder + '/' + os.listdir('../data' + '/' + folder)[0]
                data[i, :] = cv2.imread(file_).flatten()
                labels[i] = label

    X = data
    y = labels

    classNames = set(y) - set([None])
    N = len(y)

    # %%
    # subtract the mean
    Xm = X - np.ones((N, 1)) * X.mean(0)

    U, S, Vh = svd(Xm, full_matrices=False)
    # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
    # of the vector V. So, for us to obtain the correct V, we transpose:
    V = Vh.T
    # Project the centered data onto principal component space
    Z = Xm @ V

    # %%
    # Indices of the principal components to be plotted
    i = 0
    j = 1

    # Plot PCA of the data
    plt.figure(figsize=(7, 7))
    plt.style.use('seaborn')
    plt.title('PCA on dataset',  fontsize=18)
    # Z = array(Z)
    for c in classNames:
        # select indices belonging to class c:
        class_mask = np.where(np.array(y) == c)
        plt.plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
    plt.legend(classNames)
    plt.xlabel('PC{0}'.format(i + 1),  fontsize=16)
    plt.ylabel('PC{0}'.format(j + 1), fontsize=16)
    plt.savefig('PCA.png');
    plt.show()


if __name__ == "__main__":
    catalog = dict_from_json('../catalog.json')
    number_per_category_plot(catalog)
    PCAplot(catalog)
    pass
