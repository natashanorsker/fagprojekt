from utilities import dict_from_json
import json

catalog = dict_from_json('../catalog_OG.json')
has_no_class = dict_from_json('id_not_in_masterfile.json')

for product in has_no_class:
    if product in catalog.keys():
        catalog.pop(product)

a_file = open("../catalog.json", "w")
json.dump(catalog, a_file)
a_file.close()