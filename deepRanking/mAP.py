#%%
import sys
sys.path.append('..')
import numpy as np
import os
import numpy as np
import torch
import concurrent.futures
from sklearn import preprocessing

from utilities import dict_from_json, labels_from_ids, sublabels_from_ids, labels_and_metals_from_ids
from dataset import make_dataset
from nets import EmbeddingNet
from plots import extract_embeddings

np.random.seed(42069)
#%%
# labels of ids
catalog = dict_from_json('../catalog.json')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(catalog.keys()))

def ids_from_ids(ids):
    '''dummy function to make it similar to the other *_from_ids'''
    return ids

label_encoder2 = preprocessing.LabelEncoder()
#what to match against
subcategories = list(set(sublabels_from_ids(list(catalog.keys()), '../data_code/masterdata.csv')))
categories = list(set(labels_from_ids(list(catalog.keys()), '../data_code/masterdata.csv')))
labels_and_metals = list(set(labels_and_metals_from_ids(list(catalog.keys()), '../data_code/masterdata.csv')))
ids = ids_from_ids(list(catalog.keys()))

label_encoder2.fit(ids) # change categories here

K = 20

#select dataset
#make the 'normal' datasets:
train_set, test_set = make_dataset(label_encoder, n_val_products=170, NoDuplicates=False)

# where do we want to search?
# dataset = torch.utils.data.ConcatDataset([test_set])
dataset = torch.utils.data.ConcatDataset([train_set, test_set])

#make the dataloaders:
data_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)
data_loader_test = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False)

models = os.listdir('models')
models.append('vae')
models.append('random')

#%%
def main(mod):
    # print('Getting embeddings')
    if mod == 'vae' or mod == 'random':
        train_embeddings = np.load(os.path.join('../','autoencoder', "models", 'final_model', "train_embeddings.npy"))
        train_labels = np.load(os.path.join('../','autoencoder', "models", 'final_model', "train_labels.npy"), allow_pickle=True)
        test_embeddings = np.load(os.path.join('../','autoencoder', "models", 'final_model', "test_embeddings.npy"))
        test_labels = np.load(os.path.join('../','autoencoder', "models", 'final_model', "test_labels.npy"), allow_pickle=True)
        
        # the loading just gives paths, instead get actual labels
        test_labels = [i.split(os.sep)[-2] for i in test_labels]
        train_labels = [i.split(os.sep)[-2] for i in train_labels]

        all_embeddings = np.concatenate((train_embeddings, test_embeddings),axis=0)
        all_labels = np.concatenate((train_labels, test_labels),axis=0)
    else:
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
        if mod == 'vae' or mod == 'random':
            emb_label = test_labels[i]
        else:
            emb_label = label_encoder.inverse_transform([test_labels[i]])[0]

        labelq = ids_from_ids([emb_label]) # change labels here
        dists = np.sum((all_embeddings - embedding) ** 2, axis=1)
        closest_ids = np.argsort(dists)[:K*40]
        if mod == 'vae': # @k
            idx = all_labels[closest_ids]
            idx = idx[:K]
            transform = idx
            
        elif mod == 'random':
            closest_ids = np.random.choice(all_labels, 20)
            transform = closest_ids
        else:
            idx = list(set([dataset[k][1] for k in closest_ids]))
            idx = idx[:K]
            transform = label_encoder.inverse_transform(idx)

        p = np.zeros(K)
        r = np.zeros(K)

        y_true = label_encoder2.transform(labelq)
        y_pred = ids_from_ids(transform) # change to sublabels here
        y_pred = label_encoder2.transform(np.array(y_pred).ravel())

        # k ranking
        for k in range(1, K+1):
            tp = np.sum((y_true == y_pred[:k]))

            p[k-1] = tp/len(y_pred[:k])
            # fraction of objects predicted to be positive among all positive objects
            r[k-1] = tp/K
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
        np.savez('map_npz/ids/' + mod, rss=rss, cmcs=cmcs)
    except FileNotFoundError:
        os.mkdir('map_npz/ids')
        np.savez('map_npz/ids/' + mod, rss=rss, cmcs=cmcs)

# %%
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(main, models)

# single
# main(models[-1])
