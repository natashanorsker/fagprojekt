#%%
import sys
sys.path.append('..')

import numpy as np

import cv2
import numpy as np
import torch
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
from PIL import Image
from sklearn import preprocessing

from utilities import dict_from_json, labels_from_ids
from dataset import make_dataset
from nets import EmbeddingNet
from plots import extract_embeddings
from autoencoder.data_generator import get_train_test_split_paths

np.random.seed(42069)

print('Getting embeddings')
model = EmbeddingNet()
model.load_state_dict(torch.load('models/online_semi_model_margin_0.1_2021-06-13_0.0487loss.pth', map_location=torch.device('cpu')))

catalog = dict_from_json('../catalog.json')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(catalog.keys()))

#make the 'normal' datasets:
train_dataset, test_dataset = make_dataset(label_encoder, n_val_products=500, NoDuplicates=True)

dataset = torch.utils.data.ConcatDataset([test_dataset])
# dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

#make the dataloaders:
data_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)

embeddings, labels = extract_embeddings(data_loader, model)

#%%
label_encoder2 = preprocessing.LabelEncoder()
label_encoder2.fit(['Bracelets', 'Charms', 'Jewellery spare parts', 'Necklaces & Pendants', 'Rings', 'Earrings', 'Misc'])
#%%
K = 19 # number of retrieved items to query image
cmc = np.zeros(K) # @k
aps = []
for i, embedding in enumerate(embeddings):
    # query
    emb_label = label_encoder.inverse_transform([labels[i]])[0]
    labelq = labels_from_ids([emb_label])
    dists = np.sum((embeddings - embedding) ** 2, axis=1)
    closest_ids = np.argsort(dists)[:K] # @k

    transform = []  
    p = np.zeros(K)
    for j in closest_ids:
        recs = torch.tensor(dataset[j][0].numpy())
        transform.append(str(label_encoder.inverse_transform([dataset[j][1]])[0]))

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

    # binarize predictions
    y_pred[y_pred != y_true] = 0
    y_pred[y_pred == y_true] = 1
    
    ap = 1/y_pred.sum() * (p @ y_pred)

    aps.append(ap)

maP = np.mean(aps)
cmc = cmc / np.max(cmc)

# %%
print(f'mAP @ k={K}:\t', round(maP*100,2))
# rank-1
print('cmc at rank-1: \t ', cmc[0])
# rank-5
print('cmc at rank-5: \t ', cmc[4])
# CMC Curve: Plots TPIR against ranks
plt.figure()
plt.plot(range(1,K+1),cmc)
plt.xticks(range(1,K+1))
plt.xlabel('Rank')
plt.ylabel('Identification Accuracy')
plt.title('CMC Curve')
plt.ylim(0,1.02)
plt.show()
# %%
