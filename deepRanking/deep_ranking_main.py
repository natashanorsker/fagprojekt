import torch
from torchvision import transforms
from dataset import make_dataset, TripletDataset
from deep_ranking_utils import EmbeddingNet, TripletNet, TripletLoss, fit, extract_embeddings, plot_embeddings
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
cuda = torch.cuda.is_available()
import matplotlib
import matplotlib.pyplot as plt

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

#make the 'normal' datasets:
train_dataset, test_dataset, label_encoder = make_dataset()

#make triplet dataset
triplet_train_set = TripletDataset(train_dataset, train=True)
triplet_test_set = TripletDataset(test_dataset, train=False)

#make the dataloaders:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_set, batch_size=500, shuffle=True)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_set, batch_size=500, shuffle=False)

margin = 1
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 1
log_interval = 100


fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, start_epoch=0)

#train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
#plot_embeddings(train_embeddings_tl, train_labels_tl)
val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_tl, val_labels_tl)
