#%%
import numpy as np
from sklearn.metrics import top_k_accuracy_score, recall_score, average_precision_score
import sys

from torchvision.transforms import transforms
sys.path.append('..')

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sklearn import preprocessing
import torch.utils.data as data_utils


from dataset import make_dataset
from nets import EmbeddingNet
from plots import extract_embeddings
from utilities import dict_from_json, labels_from_ids

print('Getting embeddings')
model = EmbeddingNet()
model.load_state_dict(torch.load('models/online_model7-6_0.3979loss.pth', map_location=torch.device('cpu')))

catalog = dict_from_json('../catalog.json')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(catalog.keys()))

#make the 'normal' datasets:
train_dataset, test_dataset = make_dataset(label_encoder, n_test_products=10, NoDuplicates=True)

dataset = torch.utils.data.ConcatDataset([test_dataset])
# dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

#make the dataloaders:
data_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)

embeddings, labels = extract_embeddings(data_loader, model)

#%%
label_encoder2 = preprocessing.LabelEncoder()

#%%
recall = np.zeros(len(embeddings))
average_precisions = np.zeros(len(embeddings))
K = 50 # number of retrieved items
cmc = np.zeros(K) # @k
y_pred = []
for i, embedding in enumerate(embeddings):
    # query
    emb_label = label_encoder.inverse_transform([labels[i]])[0]
    labelq = labels_from_ids([emb_label])
    dists = np.sum((embeddings - embedding) ** 2, axis=1)
    closest_ids = np.argsort(dists)[:K] # @k

    transform = []  
    for j in closest_ids:
        recs = torch.tensor(dataset[j][0].numpy())
        
        transform.append(str(label_encoder.inverse_transform([dataset[j][1]])[0]))
    y_pred.append(labels_from_ids(transform))
tmp = label_encoder.inverse_transform(labels)
y_true = labels_from_ids(tmp)

label_encoder2.fit(['Bracelets', 'Charms', 'Jewellery spare parts', 'Necklaces & Pendants', 'Rings', 'Earrings'])
y_true = label_encoder2.transform(labelq)
y_pred = label_encoder2.transform(np.array(y_pred).ravel())
#%%
cmc = np.zeros(K)
for k in range(K):
    # True Positive Identification Rate (TPIR): 
    # Probability of observing the correct identity within the top K ranks
    # CMC Curve: Plots TPIR against ranks
    cmc[k] = np.sum((y_true == y_pred)[:k])/len((y_true == y_pred)[:k])


# recall = recall_score(y_true, y_pred)
# average_precision_score(y_true, y_pred)
# top_k_accuracy_score(y_true, y_pred, k=2)
# %%
plt.figure()
plt.plot(cmc)
plt.show()
# %%
