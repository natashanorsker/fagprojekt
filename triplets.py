
from data_loader import *


def sort_by_category(catalog, categories = {'ring': None, 'necklace': None, 'charm': None, 'earring': None, 'bracelet': None, 'misc': None}):
    """
    :param catalog:
    :param categories: list of categories eg. ['ring', 'necklace']
    :return:
    """
    not_found = []

    for key in list(categories.keys())[:-1]:  # don't include misc category key
        idx = []
        for id in catalog.keys():
            found = False
            if key in catalog[id]['product_name'].lower().split(' '):
                idx.append(id)
                found=True

            if not found:
                not_found.append(id)

        categories[key] = idx

    categories['misc'] = not_found

    return categories






# different categories
catalog = dict_from_json()
tester = sort_by_category(catalog=catalog)

