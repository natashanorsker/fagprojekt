from utilities import *
import os
import json
import requests
from tqdm import tqdm
import glob
import numpy as np
from numpy import expand_dims
from numpy.random import seed
import PIL
from keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array, array_to_img
from matplotlib import pyplot
import random
from PIL import Image, ImageOps
from nltk.stem import PorterStemmer

ps = PorterStemmer()

headers = ['Mozilla/5.0 CK={} (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
           'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
           'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148']


def data_retriever(directory_path, catalog):
    '''
    uses the 'product_image_url'-s from the catalog to download all the product images for every product in the catalog.
    Creates a main directory 'data' and subdirectories for every product to store the images.
    :param directory_path: The directory-path where the 'data'-directory should be created
    :param catalog: The catalog of all the products (can be found on github or can be created by running the image_scraper.py file)
    :return: directories with product images
    '''

    s = requests.Session()
    s.headers['User-Agent'] = random.choice(headers)
    s.headers['Connection'] = 'keep-alive'
    s.max_redirects = 200

    # Parent Directory path (make a DATA directory for storing the data)
    data_dir = os.path.join(directory_path, 'data')

    try:
        os.chdir(data_dir)
    except:
        os.mkdir(data_dir)
        os.chdir(data_dir)

    if not os.path.exists('model_images'):
        os.makedirs('model_images')

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
        all_items.set_postfix_str(f'Downloading a heck of a lot of images to your computer ({product})')
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
                    im = Image.open(s.get(img_urls[i], stream=True).raw)
                    mean_pixel = np.mean(img_to_array(im))
                    if mean_pixel < 200:
                        # save images with models on in another folder
                        im.save(os.path.join(data_dir, "model_images\\{}_{}_OG.jpg".format(product, str(i).zfill(2))))

                    else:
                        im_cropped = trim(im)
                        # pad and resize images:
                        im_rs = ImageOps.pad(image=im_cropped, size=(96, 96), color=im.getpixel((0, 0)))
                        im_rs.save("{}_{}_OG.jpg".format(product, str(i).zfill(2)))


                except requests.ConnectionError:
                    raise Exception(
                        'No internet connected. Try running the script when you have access to the internet.')

                except PIL.UnidentifiedImageError:
                    text_file = open('../Not_found_imgs.txt', "a")
                    text_file.write("Product: {},  URL:  {} \n".format(product, img_urls[i]))
                    text_file.close()
                    pass

            products_done[product] = True
            with open("../retrieved.txt", "w") as f:
                json.dump(products_done, f)

    # change back to the original directory:
    os.chdir(directory_path)


def rotated_image_generator(directory_path, rotation_range=180, total_images=40,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0,
                            zoom_range=0,
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
    # figure out what to call the new images (ie product_07.jpg)
    name_suffix = len(imgs)

    if name_suffix >= 40:
        return

    assert len(imgs) > 0, 'Folder at {} contains no images! :('.format(directory_path)

    product = imgs[0][:-10]  # since the file is always saved as "productiD..._02_OG.jpeg.."

    imgs_to_array = [img_to_array(load_img(x)) for x in imgs]

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


def sort_by_category(catalog,
                     categories={'ring': [], 'necklace': [], 'charm': [], 'earring': [], 'bracelet': [], 'misc': []}):
    categories_list = list(categories.keys())
    stem_categories = [ps.stem(token) for token in categories_list]

    for id in catalog.keys():
        found = False
        word_list = catalog[id]['product_name'].lower().split(' ')
        stemmed_words = [ps.stem(token) for token in word_list]

        for i in range(len(stem_categories)):
            if stem_categories[i] in stemmed_words:
                categories[categories_list[i]].append(id)
                found = True
                break

            elif 'bangl' in stemmed_words:
                categories['bracelet'].append(id)
                found = True
                break

            elif 'pendant' in stemmed_words:
                categories['necklace'].append(id)
                found = True
                break

        if not found:
            categories['misc'].append(id)
    return categories


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
        sub_dir_list.set_postfix_str(f'Creating a heck of a loads of images on your computer ({sub_dir})')
        path = os.path.join(data_dir, sub_dir)
        rotated_image_generator(path)
