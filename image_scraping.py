#from John Watson Rooney: https://www.youtube.com/watch?v=nCuPv3tf2Hg&t=84s

# imports
import requests
from bs4 import BeautifulSoup

baseurl = "https://uk.pandora.net/en/jewellery/"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
}

r = requests.get(
    "https://uk.pandora.net/en/jewellery/charms/")
soup = BeautifulSoup(r.content, 'lxml')

productList = soup.find_all('div', class_='product-tile')

productLinks = []

for item in productList:
    for link in item.find_all('a', href=True):
        print(link)
