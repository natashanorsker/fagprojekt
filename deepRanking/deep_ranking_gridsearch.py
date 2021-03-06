import sys
sys.path.append('..')

from utilities import labels_from_ids
from dataset import BalancedBatchSampler, make_dataset
from nets import EmbeddingNet
from plots import extract_embeddings, plot_embeddings
from losses import OnlineTripletLoss, AverageNonzeroTripletsMetric
from deep_ranking_utils import HardestNegativeTripletSelector, \
    SemihardNegativeTripletSelector, \
    RandomNegativeTripletSelector, Experiment
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from sklearn import preprocessing
from utilities import dict_from_json
from sklearn.model_selection import ParameterGrid
import torch
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset management and creates mini batches
from datetime import date   

cuda = torch.cuda.is_available()
#print('device:', str(torch.cuda.get_device_name()))
# PARAMETERS TO SEARCH:
param_grid = {'n_epochs': [1, 10, 15, 25], 'lr': [0.0001, 0.005, 0.1],'margin':[1]}

# PARAMETERS THAT CAN BE MANUALLY ADJUSTED:
# datasets:
n_test_products = 200 # the amount of products that goes into the test dataset
n_train_classes = 40  # the amount of products per batch in the balancedbatch sampler in the train dataloader
n_test_classes = 40  # the amount of products per batch in the balancedbatch sampler in the test dataloader
n_samples = 20

# model training:
#margin = 1  # can't go into search?

# MAKING THE DATASETS
# fit the encoder:
label_encoder = preprocessing.LabelEncoder()
catalog = dict_from_json('../catalog.json')
label_encoder.fit(list(catalog.keys()))

# make the 'normal' datasets:
train_dataset, test_dataset = make_dataset(label_encoder, n_val_products=n_test_products)

# make the batch samplers:
train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=n_train_classes, n_samples=n_samples)
test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=n_test_classes, n_samples=n_samples)

# make the dataloaders:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=n_train_classes*n_samples, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n_test_classes*n_samples, shuffle=False)

# load the dataset:
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)


# run experiments:
experiments = []

for experiment in list(ParameterGrid(param_grid)):
    # make the model:
    embedding_net = EmbeddingNet()
    model = embedding_net

    # make the sampling methods:
    random = RandomNegativeTripletSelector(experiment['margin'])
    semi_hard = SemihardNegativeTripletSelector(experiment['margin'])
    hard = HardestNegativeTripletSelector(experiment['margin'])

    for sampling_method in [random, semi_hard, hard]:

        # HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
        loss_fn = OnlineTripletLoss(experiment['margin'], sampling_method)
        optimizer = optim.Adam(model.parameters(), lr=experiment['lr'], weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

        if cuda:
            model.cuda()

        # make the whole grid thing here
        run = Experiment(train_loader=online_train_loader, val_loader=online_test_loader, model=model, label_encoder=label_encoder, loss_fn=loss_fn,
                         optimizer=optimizer, scheduler=scheduler, cuda=cuda, kind=sampling_method.name,
                         to_tensorboard=True, metrics=[AverageNonzeroTripletsMetric()], start_epoch=0, margin=experiment['margin'], lr=experiment['lr'],
                         n_epochs=experiment['n_epochs'])

        experiments.append(run)
        torch.save(run.model.state_dict(), 'models/online_{}_model_ep_{}_lr_{}_margin_{}_date_{}_loss_{}.pth'.format(sampling_method.name, experiment['n_epochs'], experiment['lr'], experiment['margin'], date.today(),round(run.val_loss, 4)))
