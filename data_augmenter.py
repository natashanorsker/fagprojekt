from numpy import expand_dims
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import random


def rotated_image_generator(directory_path, rotation_range = 90, total_images=40, save=True, show=False):
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
    imgs = os.listdir()
    product = imgs[1][:-7]

    # ImageDataGenerator rotation
    datagen = ImageDataGenerator(rotation_range=rotation_range, fill_mode='nearest')

    for i in range(len(imgs), total_images):  # we want 40 images per product
        # choose at random from images in folder:
        img_path = random.choice(imgs)

        # specify name of image to save as
        img_name = product + '_' + str(i).zfill(2) + '_AU.jpg'

        img = load_img(random.choice(imgs))

        # convert to numpy array
        data = img_to_array(img)

        # iterator
        aug_iter = datagen.flow(expand_dims(data, 0), batch_size=1)

        # generate batch of images
        image = next(aug_iter)[0].astype('uint8')

        if save:
            save_img(img_name, image)

        if show:
            # plot raw pixel data
            pyplot.imshow(image)





rotated_image_generator('C:\\Users\\natas\\OneDrive\\Skrivebord\\DTU\\Fagprojekt\\fagprojekt\\data\\168742C00', show=True)