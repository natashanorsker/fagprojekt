from dataset import BalancedBatchSampler, make_dataset
from nets import EmbeddingNet
from plots import extract_embeddings, plot_embeddings
from losses import OnlineTripletLoss, AverageNonzeroTripletsMetric
from deep_ranking_utils import AllTripletSelector, HardestNegativeTripletSelector, \
    SemihardNegativeTripletSelector, \
    RandomNegativeTripletSelector, Experiment
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from sklearn import preprocessing
from utilities import dict_from_json
from sklearn.model_selection import ParameterGrid
import torch
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import (DataLoader,)  # Gives easier dataset managment and creates mini batches

cuda = torch.cuda.is_available()

# PARAMETERS TO USE:
n_epochs = 1
lr = 0.001

# PARAMETERS THAT CAN BE MANUALLY ADJUSTED:
# datasets:
n_test_products = 100 # the amount of products that goes into the test dataset (sat very high for debugging)
n_train_classes = 40  # the amount of products per batch in the balancedbatch sampler in the train dataloader
n_test_classes = 80  # the amount of products per batch in the balancedbatch sampler in the test dataloader
n_samples = 10

# model training:
margin = 1  # can't go into search?

# MAKING THE DATASETS
# fit the encoder:
label_encoder = preprocessing.LabelEncoder()
catalog = dict_from_json('../catalog.json')
label_encoder.fit(list(catalog.keys()))

# make the 'normal' datasets:
train_dataset, test_dataset = make_dataset(label_encoder, n_test_products=n_test_products)

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

# make the model:
embedding_net = EmbeddingNet()
model = embedding_net
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)


# make the whole grid thing here
run = Experiment(train_loader=online_train_loader, val_loader=online_test_loader, model=model, loss_fn=loss_fn,
                 optimizer=optimizer, scheduler=scheduler, cuda=cuda,
                 to_tensorboard=True, metrics=[AverageNonzeroTripletsMetric()], start_epoch=0, margin=margin,
                 lr=lr,
                 n_epochs=n_epochs)



# SAVING THE MODEL:
# save model:
torch.save(run.model.state_dict(), 'production_models/online_model_{}loss.pth'.format(round(run.val_loss, 4)))

# load model:
#the_model = EmbeddingNet()
#the_model.load_state_dict(torch.load('online_model.pth'))

# extract embeddings and plot:
#val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, the_model)
#plot_embeddings(val_embeddings_tl, val_labels_tl, encoder=label_encoder)