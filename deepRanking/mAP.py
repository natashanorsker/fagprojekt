#%%
import sys
sys.path.append('..')
import numpy as np
import os
import pathlib
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
train_set, test_set = make_dataset(label_encoder, n_val_products=100, NoDuplicates=False)

# where do we want to search?
# dataset = torch.utils.data.ConcatDataset([test_set])
dataset = torch.utils.data.ConcatDataset([train_set, test_set])

#make the dataloaders:
data_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)
data_loader_test = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False)

label_encoder2 = preprocessing.LabelEncoder()
df = pd.read_csv('data_code/masterdata.csv', sep=';') 
df.columns = df.columns.str.lower()
subcategories = list(set(df['item sub-category']))
categories = ['Bracelets', 'Charms', 'Jewellery spare parts', 'Necklaces & Pendants', 'Rings', 'Earrings', 'Misc']
label_encoder2.fit(subcategories)

models = os.listdir('models')

#%%
def main(mod):
    # print('Getting embeddings')
    model = EmbeddingNet()
    mpath = 'models/' + mod
    model.load_state_dict(torch.load(mpath, map_location=torch.device('cpu')))

    all_embeddings, all_labels = extract_embeddings(data_loader, model)
    test_embeddings, test_labels = extract_embeddings(data_loader_test, model)

    #%%
    K = 20 # number of retrieved items to query image
    cmc = np.zeros(K) # @k
    aps = []
    for i, embedding in enumerate(test_embeddings):
        # query
        emb_label = label_encoder.inverse_transform([test_labels[i]])[0]
        labelq = sublabels_from_ids([emb_label])
        dists = np.sum((all_embeddings - embedding) ** 2, axis=1)
        closest_ids = np.argsort(dists)[:K*40] # @k
        idx = list(set([dataset[k][1] for k in closest_ids]))
        idx = idx[:K]
        transform = label_encoder.inverse_transform(idx)

        p = np.zeros(K)

        y_true = label_encoder2.transform(labelq)
        y_pred = labels_from_ids(transform)
        y_pred = label_encoder2.transform(np.array(y_pred).ravel())

        for k in range(1, K+1):
            p[k-1] = np.sum((y_true == y_pred[:k]))/len(y_pred[:k])

            # True Positive Identification Rate (TPIR): 
            # Probability of observing the correct identity within the top K ranks
            if y_true in y_pred[:k]:
                t = np.where(y_pred==y_true)[0][0]
                cmc[t:] += 1 

            # TODO: add calculation for recall @ k curves

        # binarize predictions
        y_pred[y_pred != y_true] = 0
        y_pred[y_pred == y_true] = 1
        
        ap = 1/(y_pred.sum() + 1e-6) * (p @ y_pred)

        aps.append(ap)

    maP = np.mean(aps)
    cmc = cmc / np.max(cmc)

    print(f'model: {mod}')
    print(f'mAP @ k={K}:\t', round(maP*100,2))
    # rank-1
    print('cmc at rank-1: \t ', round(cmc[0]*100,2))
    # rank-5
    print('cmc at rank-5: \t ', round(cmc[4]*100,2))
    # CMC Curve: Plots TPIR against ranks
    plt.figure()
    plt.plot(range(1,K+1),cmc)
    plt.xticks(range(1,K+1))
    plt.xlabel('Rank')
    plt.ylabel('Identification Accuracy')
    plt.title('CMC Curve')
    plt.ylim(0,1.02)
    plt.savefig(f'../Figures/subcategorycmccurve{mod[:-23]}.png',dpi=200)
    plt.show()

# %%
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(main, models)
    