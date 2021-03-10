# imports
import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm


def file_from_json(path="catalog.json"):
    # open the product catalog:
    a_file = open(path, "r")
    catalog = json.loads(a_file.read())
    a_file.close()

    return catalog


def create_catalog(save=True):
    headers = {'User-Agent': 'Mozilla/5.0 CK={} (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
               "Connection": 'keep-alive'}

    new_catalog = {}
    product_urls = {}

    # get info on all products and their primary image url:
    for page in range(0, 1080, 36):
        j = requests.get("https://uk.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element".format(page))
        jewellery = BeautifulSoup(j.content, 'lxml')

        product_list = jewellery.find_all('input', class_='product-details', attrs={"value": True})

        for item in product_list:
            info_dict = json.loads(item['value'])
            new_catalog[info_dict["product_id"]] = info_dict
            product_urls[info_dict["product_id"]] = info_dict['product_url']

    # get all images for every product:
    for iD, url in tqdm(product_urls.items()):
        site = requests.get(url)
        product = BeautifulSoup(site.content, 'lxml')

        # list of image_urls
        image_list = product.find_all('a', class_='main-image', attrs={'href': True})

        url_list = []

        for item in image_list:
            url_list.append(item['href'])

        # find all the spin360 images:
        spin360 = product.find_all('img', class_='spin-reel', attrs={'data-frames': True, 'data-images': True})

        if spin360:
            # get every other spin360 image (all of them are too much)
            for frame in range(1, int(spin360[0]['data-frames']), 2):
                i = str(frame).zfill(2)
                url_list.append(spin360[0]['data-images'].replace("##", i))

        new_catalog[iD]['product_image_url'] = url_list

    if save:
        a_file = open("catalog.json", "w")
        json.dump(new_catalog, a_file)
        a_file.close()

    return new_catalog
