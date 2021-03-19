# imports
import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm


#websites: (these are the websites with the same format as UK)
websites = ['https://us.pandora.net/en/jewelry/#position=top&src=categorySearch',
            'https://au.pandora.net/en/jewellery/#position=top&src=categorySearch',
            'https://cn.pandora.net/zh/jewellery/#position=top&src=categorySearch',
            'https://hk.pandora.net/en/jewellery/#position=top&src=categorySearch',
            'https://jp.pandora.net/ja/jewellery/#position=top&src=categorySearch',
            'https://hk.pandora.net/en/jewellery/#position=top&src=categorySearch',
            'https://nz.pandora.net/en/jewellery/#position=top&src=categorySearch',
            'https://sg.pandora.net/en/jewellery/#position=top&src=categorySearch',
            'https://dk.pandora.net/da/smykker/#position=top&src=categorySearch',
            'https://de.pandora.net/de/schmuck/#position=top&src=categorySearch',
            'https://fr.pandora.net/fr/bijoux/#position=top&src=categorySearch',
            'https://it.pandora.net/it/gioielli/#position=top&src=categorySearch',
            'https://uk.pandora.net/en/jewellery/#position=top&src=categorySearch',
            'https://nl.pandora.net/nl/sieraden/#position=top&src=categorySearch',
            'https://pl.pandora.net/pl/bizuteria/#position=top&src=categorySearch',
            'https://se.pandora.net/sv/smycken/#position=top&src=categorySearch',
            'https://at.pandora.net/de/schmuck/#position=top&src=categorySearch']

websites_edited = ['https://us.pandora.net/en/jewelry/?start={}&amp;sz=36&amp;format=page-element',
            'https://au.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element',
            'https://cn.pandora.net/zh/jewellery/?start={}&amp;sz=36&amp;format=page-element',
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
            'https://at.pandora.net/de/schmuck/?start={}&amp;sz=36&amp;format=page-element']


def create_catalog(save=True):
    headers = {'User-Agent': 'Mozilla/5.0 CK={} (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
               "Connection": 'keep-alive'}

    new_catalog = {}
    product_urls = {}

    # get info on all products and their primary image url:
    for page in range(0, 1150, 36):
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

if __name__ == "__main__":
    new_catalog = create_catalog(save=True)
