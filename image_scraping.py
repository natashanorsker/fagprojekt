
# imports
import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

headers = {
    'User-Agent': 'Mozilla/5.0 CK={} (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    "Connection": 'keep-alive'
}

catalog = {}
product_urls = {}

# get info on all products and their primary image url:
for page in range(0, 1080, 36):
    j = requests.get("https://uk.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element".format(page))
    jewellery = BeautifulSoup(j.content, 'lxml')

    productList = jewellery.find_all('input', class_='product-details', attrs={"value": True})

    for item in productList:
        info_dict = json.loads(item['value'])
        catalog[info_dict["product_id"]] = info_dict
        product_urls[info_dict["product_id"]] = info_dict['product_url']


# get all images for every product:
for iD, url in tqdm(product_urls.items()):
    site = requests.get(url)
    product = BeautifulSoup(site.content, 'lxml')

    # list of image_urls
    imageList = product.find_all('a', class_='main-image', attrs={'href': True})

    url_list = []

    for item in imageList:
        url_list.append(item['href'])

    spin360 = product.find_all('img', class_='spin-reel', attrs={'data-frames': True, 'data-images': True})

    if spin360:
        # get all the spin360 images
        for frame in range(1, int(spin360[0]['data-frames'])):
            i = str(frame).zfill(2)
            url_list.append(spin360[0]['data-images'].replace("##", i))

    catalog[iD]['product_image_url'] = url_list


#save the catalog Dict:
a_file = open("catalog.json", "w")
json.dump(catalog, a_file)
a_file.close()

'''
#to open the catalog (also in another script):
a_file = open("catalog.json", "r")
catalog = json.loads(a_file.read())
a_file.close()
'''
