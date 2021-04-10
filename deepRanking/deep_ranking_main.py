import torch
from torchvision import transforms
from dataset import make_dataset
from deep_ranking_utils import EmbeddingNet, TripletNet, TripletLoss, fit
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

#make the datasets:
triplet_train, triplet_test, label_encoder = make_dataset()

#make the dataloaders:
triplet_train_loader = torch.utils.data.DataLoader(triplet_train, batch_size=128, shuffle=True)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test, batch_size=128, shuffle=False)

margin = 1.
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100


fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


