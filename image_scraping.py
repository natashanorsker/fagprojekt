# imports
import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import random


#websites: (these are the websites with the same format as UK)
#does not work:
#'https://us.pandora.net/en/jewelry/?start={}&amp;sz=36&amp;format=page-element''

websites = ['https://cn.pandora.net/zh/jewellery/?start={}&amp;sz=36&amp;format=page-element',
            'https://hk.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element',
            'https://jp.pandora.net/ja/jewellery/?start={}&amp;sz=36&amp;format=page-element',
            'https://hk.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element',
            'https://nz.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element',
            'https://sg.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element',
            'https://dk.pandora.net/da/smykker/?start={}&amp;sz=36&amp;format=page-element',
            'https://de.pandora.net/de/schmuck/?start={}&amp;sz=36&amp;format=page-element',
            'https://fr.pandora.net/fr/bijoux/?start={}&amp;sz=36&amp;format=page-element',
            'https://it.pandora.net/it/gioielli/?start={}&amp;sz=36&amp;format=page-element',
            'https://uk.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element',
            'https://nl.pandora.net/nl/sieraden/?start={}&amp;sz=36&amp;format=page-element',
            'https://pl.pandora.net/pl/bizuteria/?start={}&amp;sz=36&amp;format=page-element',
            'https://se.pandora.net/sv/smycken/?start={}&amp;sz=36&amp;format=page-element',
            'https://at.pandora.net/de/schmuck/?start={}&amp;sz=36&amp;format=page-element',
            'https://au.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element',
            'https://au.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element']

headers = ['Mozilla/5.0 CK={} (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
           'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148']

def create_catalog(websites, save=True):

    new_catalog = {}


    for site in tqdm(websites, desc='For every site'):
        s = requests.Session()
        s.headers['User-Agent'] = random.choice(headers)
        s.max_redirects = 60

        product_urls = {}

        #just get the start site in order to look up total products on the site
        a = s.get(site.format('0'))
        start = BeautifulSoup(a.content, 'lxml')
        total_products = int(start.find('input', id='products_count')['value'])
        products_per_page = int(start.find('input', id='pageload_product_count')['value'])

        # get info on all products and their primary image url:
        for page in range(0, total_products, products_per_page):
            j = s.get(site.format(page))
            jewellery = BeautifulSoup(j.content, 'lxml')

            product_list = jewellery.find_all('input', class_='product-details', attrs={"value": True})

            for item in product_list:
                info_dict = json.loads(item['value'])
                if info_dict["product_id"] in new_catalog:
                    pass
                else:
                    new_catalog[info_dict["product_id"]] = info_dict
                    product_urls[info_dict["product_id"]] = info_dict['product_url']

        # get all images for every product:
        for iD, url in tqdm(product_urls.items(), desc='Collecting all new image urls', leave='False'):
            product_site = s.get(url)
            product = BeautifulSoup(product_site.content, 'lxml')

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

if __name__ == "__main__":
    new_catalog = create_catalog(save=True, websites=websites)
