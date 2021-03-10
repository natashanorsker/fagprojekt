import os
import json
import urllib.request
import requests
from tqdm import tqdm
import glob
import re
import numpy as np
from numpy import expand_dims
from numpy.random import seed
from keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array, array_to_img
from matplotlib import pyplot
import random
from PIL import Image
#from image_scraping import file_from_json


def dict_from_json(path="catalog.json"):
    # open the product catalog:
    a_file = open(path, "r")
    catalog = json.loads(a_file.read())
    a_file.close()
    return catalog

def data_retriever(directory_path, catalog):
    '''
    uses the 'product_image_url'-s from the catalog to download all the product images for every product in the catalog.
    Creates a main directory 'data' and subdirectories for every product to store the images.
    :param directory_path: The directory-path where the 'data'-directory should be created
    :param catalog: The catalog of all the products (can be found on github or can be created by running the image_scraper.py file)
    :return: directories with product images
    '''

    # Parent Directory path (make a DATA directory for storing the data)
    data_dir = os.path.join(directory_path, 'data')

    try:
        os.chdir(data_dir)
    except:
        os.mkdir(data_dir)
        os.chdir(data_dir)

    # file that stores products already looked up
    if os.path.isfile('retrieved.txt'):
        with open('retrieved.txt', 'r') as f:
            products_done = json.load(f)
    else:
        products_done = {}
        with open("retrieved.txt", "w") as f:
            json.dump(products_done, f)

    all_items = tqdm(catalog.items())
    for product, info in all_items:
        all_items.set_postfix_str('Downloading a heck of a lot of images to your computer')
        if product not in products_done.keys():

            new_dir = os.path.join(data_dir, product)
            try:
                os.chdir(new_dir)
            except:
                os.mkdir(new_dir)
                os.chdir(new_dir)

            img_urls = info['product_image_url']

            for i in range(len(img_urls)):
                try:
                    im = Image.open(requests.get(img_urls[i], stream=True).raw)
                    im_rs = im.resize((200,200))
                    im_rs.save("{}_{}.jpg".format(product, str(i).zfill(2)))
                    #urllib.request.urlretrieve(img_urls[i], "{}_{}.jpg".format(product, str(i).zfill(2)))

                except:
                    text_file = open('../Not_found_imgs.txt', "a")
                    text_file.write("Product: {},  URL:  {} \n".format(product, img_urls[i]))
                    text_file.close()
                    pass

            products_done[product] = True
            with open("../retrieved.txt", "w") as f:
                json.dump(products_done, f)

    #change back to the original directory:
    os.chdir(directory_path)



def rotated_image_generator(directory_path, rotation_range = 180, total_images=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest', save=True, show=False):
    '''
    :param directory_path: (str) the directory where the images are stored (should be the subdirectories in the 'data' directory)
    :param rotation_range: (int) the range of the image rotation (between 0-360 degrees)
    :param total_images: (int)
    :param width_shift_range:
    :param height_shift_range:
    :param shear_range:
    :param zoom_range:
    :param horizontal_flip:
    :param fill_mode:
    :param save: (bool) if to save the images to the computer or not
    :param show: (bool) if to show the images or not
    :return:
    '''
    os.chdir(directory_path)

    # get list of all jpg image files in directory
    imgs = glob.glob("*.jpg")
    #figure out what to call the new images (ie product_07.jpg)
    name_suffix = len(imgs)

    if name_suffix >= 40:
        return

    assert len(imgs) > 0, 'Folder at {} contains no images! :('.format(directory_path)

    product = imgs[0][:-7] #since the file is always saved as "productiD..._02.jpeg.."

    #imgs to array only contains catalog images that doesn't have a model in it (ie. the mean pixel value is over 200)
    imgs_to_array = [img_to_array(load_img(x)) for x in imgs if np.mean(img_to_array(load_img(x))) > 200]

    assert len(imgs_to_array) > 0, 'No images to rotate! :('

    # ImageDataGenerator rotation
    datagen = ImageDataGenerator(rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode)

    for i in range(len(imgs_to_array), total_images):  # we want 40 images per product
        # choose at random from images:
        data = random.choice(imgs_to_array)
        # specify name of image to save as
        img_name = product + '_' + str(name_suffix).zfill(2) + '_AU.jpg'

        # iterator
        aug_iter = datagen.flow(expand_dims(data, 0), batch_size=1)

        # generate batch of images
        image = next(aug_iter)[0].astype('uint8')

        name_suffix += 1

        if save:
            save_img(img_name, image)

        if show:
            # plot raw pixel data
            pyplot.imshow(image)
            pyplot.show()


if __name__ == "__main__":

    # set seed
    # important in order to get the same rotated images:
    random.seed(420)
    seed(420)

    catalog = dict_from_json("catalog.json")

    # download the product images from pandoras website:
    data_retriever(os.getcwd(), catalog)

    data_dir = os.path.join(os.getcwd(), 'data')
    sub_dir_list = tqdm(os.listdir(data_dir))

    # augment the product images so that there are 40 images per product
    # - we want to do this for every subdirectory in the 'data' directory:
    for sub_dir in sub_dir_list:
        if "." in sub_dir:
            continue
        sub_dir_list.set_postfix_str('Creating a heck of a loads of images on your computer')
        path = os.path.join(data_dir, sub_dir)
        rotated_image_generator(path)

