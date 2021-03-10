import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

headers = {
    'User-Agent': 'Mozilla/5.0 CK={} (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
    "Connection": 'keep-alive'
}

a_file = open("catalog.json", "r")
catalog = json.loads(a_file.read())
a_file.close()

product_urls = {'799364C00': 'https://uk.pandora.net/en/jewellery/charms/charms/heart-clover-charm/799364C00.html'}

for id, url in product_urls.items():
    site = requests.get(url)
    product = BeautifulSoup(site.content, 'lxml')

    more_info = product.find('ul', class_='section-content')
    info = [string for string in more_info.get_text().split('\n') if string]
    pass

    product_description = product.find('div', class_='desktop-detail-text').get_text(strip=True)


pass

