#%%
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree
from sklearn import preprocessing
from torchvision import transforms
import itertools

from deepRanking.dataset import make_dataset
from deepRanking.nets import EmbeddingNet
from deepRanking.plots import extract_embeddings
import detectron2segment.inference
from utilities import dict_from_json

#%% Step 1 get embeddings for corpus
print('Getting embeddings')
model = EmbeddingNet()
model.load_state_dict(torch.load('deepRanking/models/online_model7-6_0.3979loss.pth', map_location=torch.device('cpu')))

catalog = dict_from_json('catalog.json')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(catalog.keys()))

#make the 'normal' datasets:
train_dataset, test_dataset = make_dataset(label_encoder, n_val_products=100, NoDuplicates=True)

dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
#make the dataloaders:
data_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)

embeddings, labels = extract_embeddings(data_loader, model)

