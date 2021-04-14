import numpy as np
import os

import json
def dict_from_json(path):
    # open the product catalog:
    a_file = open(path, "r")
    catalog = json.loads(a_file.read())
    a_file.close()
    return catalog
categories = dict_from_json('./catalog_by_category.json')

filenames = []
d = 'data'
walk = os.walk(d)
for root, dirs, files in walk:
    for file in files:
        if ".jpg" in file and "checkpoint.jpg" not in file:
            filenames.append(os.path.join(root, file))

N = len(filenames)

exclude = set(['model_images'])
folders = [os.path.join(d, o)[len(d)+1:] for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o)) and (os.path.join(d,o)[len(d)+1:] not in exclude)]

labels = []
for folder in folders:
    for key, val in categories.items():
        if folder in val:
            labels.append([key] * len(os.listdir(d+'/'+folder)))
flat_labels = [item for sublist in labels for item in sublist]
            
labels = np.array(flat_labels)
print(len(labels))