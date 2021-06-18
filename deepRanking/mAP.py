#%%
import sys
sys.path.append('..')
import numpy as np
import os
import numpy as np
import torch
import concurrent.futures
from matplotlib import pyplot as plt
from sklearn import preprocessing
import pandas as pd

from utilities import dict_from_json, labels_from_ids, sublabels_from_ids
from autoencoder.train_test import get_train_test_split_paths
from dataset import make_dataset, list_paths_labels
from nets import EmbeddingNet
from plots import extract_embeddings

np.random.seed(42069)
#%%
catalog = dict_from_json('../catalog.json')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(catalog.keys()))

#make the 'normal' datasets:
train_set, test_set = make_dataset(label_encoder, n_val_products=30, NoDuplicates=False)

# where do we want to search?
# dataset = torch.utils.data.ConcatDataset([test_set])
dataset = torch.utils.data.ConcatDataset([train_set, test_set])

#make the dataloaders:
data_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)
data_loader_test = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False)

label_encoder2 = preprocessing.LabelEncoder()
df = pd.read_csv('../data_code/masterdata.csv', sep=';') 
df.columns = df.columns.str.lower()
subcategories = list(set(df['item sub-category']))
subcategories = [i.lower() for i in subcategories]
categories = ['Bracelets', 'Charms', 'Jewellery spare parts', 'Necklaces & Pendants', 'Rings', 'Earrings', 'Misc']
label_encoder2.fit(categories) # change to subcategories here

models = os.listdir('models')
K = 20
#%%
def main(mod):
    # print('Getting embeddings')
    model = EmbeddingNet()
    mpath = 'models/' + mod
    model.load_state_dict(torch.load(mpath, map_location=torch.device('cpu')))

    all_embeddings, all_labels = extract_embeddings(data_loader, model)
    test_embeddings, test_labels = extract_embeddings(data_loader_test, model)

    #%%
    # K = 20 # number of retrieved items to query image
    cmc = np.zeros(K) # @k
    aps = []
    rs = np.zeros((len(test_embeddings), K))
    for i, embedding in enumerate(test_embeddings):
        # query
        emb_label = label_encoder.inverse_transform([test_labels[i]])[0]
        labelq = labels_from_ids([emb_label]) # change to sublabels here
        dists = np.sum((all_embeddings - embedding) ** 2, axis=1)
        closest_ids = np.argsort(dists)[:K*40] # @k
        idx = list(set([dataset[k][1] for k in closest_ids]))
        idx = idx[:K]
        transform = label_encoder.inverse_transform(idx)

        p = np.zeros(K)
        r = np.zeros(K)

        y_true = label_encoder2.transform(labelq)
        y_pred = labels_from_ids(transform) # change to sublabels here
        y_pred = label_encoder2.transform(np.array(y_pred).ravel())

        # k ranking
        for k in range(1, K+1):
            tp = np.sum((y_true == y_pred[:k]))
            fn = np.sum((y_true == y_pred))

            p[k-1] = tp/len(y_pred[:k])
            # fraction of objects predicted to be positive among all positive objects
            r[k-1] = tp/(tp + fn + 1e-6)
            # True Positive Identification Rate (TPIR): 
            # Probability of observing the correct identity within the top K ranks
            if y_true in y_pred[:k]:
                t = np.where(y_pred==y_true)[0][0]
                cmc[t:] += 1 

        # binarize predictions
        y_pred[y_pred != y_true] = 0
        y_pred[y_pred == y_true] = 1
        
        ap = 1/(y_pred.sum() + 1e-6) * (p @ y_pred)

        aps.append(ap)
        rs[i, :] = r

    rss = np.mean(rs, axis=0)
    maP = np.mean(aps)
    cmc = cmc / np.max(cmc)
    cmcs = cmc

    print(f'model: {mod}')
    print(f'mAP @ k={K}, cmc at rank-1, cmc at rank-5')
    print(round(maP*100,2),'&', round(cmc[0]*100,2),'&', round(cmc[4]*100,2))

    # log
    try:
        np.savez('map_npz/categories/' + mod, rss=rss, cmcs=cmcs)
    except FileNotFoundError:
        os.mkdir('map_npz/categories/')
        np.savez('map_npz/categories/' + mod, rss=rss, cmcs=cmcs)

# %%
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(main, models)
