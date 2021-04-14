from deep_ranking_utils import plot_embeddings, EmbeddingNet, TripletNet, extract_embeddings
from dataset import make_dataset, make_plot_dataset
import torch
from sklearn import preprocessing
from utilities import dict_from_json

label_encoder = preprocessing.LabelEncoder()
catalog = dict_from_json('../catalog.json')
label_encoder.fit(list(catalog.keys()))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

#load the model
#load model:
the_model = EmbeddingNet()
the_model.load_state_dict(torch.load('online_model.pth'))

#make the datasets:
plot_dataset = make_plot_dataset(label_encoder, 10)
plot_loader = torch.utils.data.DataLoader(plot_dataset, batch_size=400, shuffle=False)

#make the 'normal' datasets:
train_dataset, test_dataset = make_dataset(label_encoder)

#make the dataloaders:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)

val_embeddings_tl, val_labels_tl = extract_embeddings(plot_loader, the_model)
val_labels_tl = val_labels_tl.astype(int)
#val_labels_tl = label_encoder.inverse_transform(list(val_labels_tl.astype(int)))
plot_embeddings(val_embeddings_tl, val_labels_tl, colors=colors, label_encoder=label_encoder)

#
#val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, the_model)
#plot_embeddings(val_embeddings_tl, val_labels_tl)


