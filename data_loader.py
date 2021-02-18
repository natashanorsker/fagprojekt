import os
import json
import urllib.request
from tqdm import tqdm
import sys
from PIL import Image
from io import BytesIO

consent = input('Warning: downloading a heck of a lot of images to your computer. Do you wish to proceed? [y/n]:  ')

if consent in ['y', 'yes', 'hell ya', 'ja', 'jeps', 'oui', 'tres bien']:

    # Parent Directory path (make a DATA directory for storing the data)
    data_dir = os.path.join(os.getcwd(), 'DATA')

    try:
        os.chdir(data_dir)
    except:
        os.mkdir(data_dir)
        os.chdir(data_dir)

    # open the product catalog:
    a_file = open("../catalog.json", "r")
    catalog = json.loads(a_file.read())
    a_file.close()


    for product, info in tqdm(catalog.items()):

        new_dir = os.path.join(data_dir, product)
        try:
            os.chdir(new_dir)
        except:
            os.mkdir(new_dir)
            os.chdir(new_dir)

        img_urls = info['product_image_url']

        for i in range(len(img_urls)):
            try:
                urllib.request.urlretrieve(img_urls[i], "{}_{}.jpg".format(product, str(i).zfill(2)))

            except:
                text_file = open('../Not_found_imgs.txt', "a")
                text_file.write("Product: {},  URL:  {}".format(product, img_urls[i]))
                text_file.close()
                pass