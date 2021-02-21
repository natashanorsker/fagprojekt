import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import random

from numpy.random import seed
random.seed(420)
seed(420)

import glob

def rotated_image_generator(directory_path, rotation_range = 180, total_images=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest', save=True, show=False):

    '''

    :param directory_path: the path to the directory you want to save images to
    :param rotation_range: the range the images can be rotated in (int in range [0-360])
    :param total_images: the total number of images you want in the folder
    :return:
    :save: whether to save the images to the directory or not
    :show: whether to show images while making them
    '''

    os.chdir(directory_path)

    # get list of all image files in directory
    imgs = glob.glob("*.jpg")
    product = imgs[1][:-7] #since the file is always saved as "productiD..._02.jpeg.."

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

    #figure out what to call the new images (ie product_07.jpeg)
    name_suffix = len(imgs)

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




rotated_image_generator('C:\\Users\\natas\\OneDrive\\Skrivebord\\Proof-of-concept data augmentation\\798072CZ', show=True)