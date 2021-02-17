#from John Watson Rooney: https://www.youtube.com/watch?v=nCuPv3tf2Hg&t=84s

# imports
import requests
import json
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
}

catalog = {}

for page in range(0, 1078, 36):
    j = requests.get("https://uk.pandora.net/en/jewellery/?start={}&amp;sz=36&amp;format=page-element".format(page))
    jewellery = BeautifulSoup(j.content, 'lxml')

    productList = jewellery.find_all('input', class_='product-details', attrs={"value": True})

    for item in productList:
        info_dict = json.loads(item['value'])
        catalog[info_dict["product_id"]] = info_dict


#save the catalog Dict:
a_file = open("catalog.json", "w")
json.dump(catalog, a_file)
a_file.close()

#to open the catalog (also in another script):
a_file = open("catalog.json", "r")
catalog = a_file.read()
a_file.close()

print('finito')