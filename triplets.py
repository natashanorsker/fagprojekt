
from data_loader import *
from nltk.stem import PorterStemmer
ps = PorterStemmer()

def sort_by_category(catalog, categories = {'ring': [], 'necklace': [], 'charm': [], 'earring': [], 'bracelet': [], 'misc': []}):
    """
    :param catalog:
    :param categories: list of categories eg. ['ring', 'necklace']
    :return:
    """
    categories_list = list(categories.keys())
    stem_categories = [ps.stem(token) for token in categories_list]

    for id in catalog.keys():
        found = False
        word_list = catalog[id]['product_name'].lower().split(' ')
        stemmed_words = [ps.stem(token) for token in word_list]


        for i in range(len(stem_categories)):
            if stem_categories[i] in stemmed_words:
                categories[categories_list[i]].append(id)
                found=True
                break

            elif 'bangl' in stemmed_words:
                categories['bracelet'].append(id)
                found=True
                break

            elif 'pendant' in stemmed_words:
                categories['necklace'].append(id)
                found=True
                break

        if not found:
            categories['misc'].append(id)
    return categories






# different categories
catalog = dict_from_json()
categories = sort_by_category(catalog=catalog)

