from dataset import BalancedBatchSampler, make_dataset
from deep_ranking_utils import EmbeddingNet,OnlineTripletLoss, AllTripletSelector, HardestNegativeTripletSelector, SemihardNegativeTripletSelector, RandomNegativeTripletSelector, AverageNonzeroTripletsMetric, fit
from deep_ranking_utils import extract_embeddings, plot_embeddings
from torch.optim import lr_scheduler
import torch.optim as optim
import torch
from torch.autograd import Variable
cuda = torch.cuda.is_available()

#make the 'normal' datasets:
train_dataset, test_dataset, label_encoder = make_dataset()

#make the batch samplers:
train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=25)
test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=25)

#
#make the dataloaders:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)

#load the dataset:
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 1
log_interval = 150

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])

#save model:
torch.save(model.state_dict(), 'online_model.pth')

#load model:
the_model = EmbeddingNet()
the_model.load_state_dict(torch.load('online_model.pth'))

#train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
#plot_embeddings(train_embeddings_otl, train_labels_otl)
val_embeddings_otl, val_labels_otl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_otl, val_labels_otl)
