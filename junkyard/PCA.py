import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

def dict_from_json(path):
    # open the product catalog:
    with open(path, "r") as a_file:
        catalog = json.loads(a_file.read())
    return catalog

categories = dict_from_json('../catalog.json')

include = set(['bracelets', 'charms', 'necklaces', 'rings', 'earrings'])
data = np.zeros((len(categories), 96*96*3),dtype='uint8')

labels = [None] * len(categories)
for i, folder in enumerate(categories.keys()):
    if folder in os.listdir('../data'):
        label = categories[folder]['product_category']
        if label in include:
            file_ = '../data'+'/'+folder+'/'+os.listdir('../data'+'/'+folder)[0]
            data[i, :] = cv2.imread(file_).flatten()
            labels[i] = label

X = data
y = labels

classNames = set(y)-set([None])
classDict = dict(zip(classNames, range(len(classNames))))
N = len(y)
C = len(classNames)

#%%
# subtract the mean 
Xm = X - np.ones((N,1))*X.mean(0)

U,S,Vh = svd(Xm, full_matrices=False) 
# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T
# Project the centered data onto principal component space
Z = Xm @ V

#%%
# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
plt.style.use('seaborn')
f = plt.figure(figsize=(6,6))
plt.title('PCA on dataset')
#Z = array(Z)
for c in classNames:
    # select indices belonging to class c:
    class_mask = np.where(np.array(y) == c)
    plt.plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1));
#plt.savefig('PCA.pdf');
plt.show()


