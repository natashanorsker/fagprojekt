from deep_ranking_utils import plot_embeddings, EmbeddingNet, TripletNet, extract_embeddings
from dataset import make_dataset, make_plot_dataset
import torch
from sklearn import preprocessing
from utilities import dict_from_json

label_encoder = preprocessing.LabelEncoder()
catalog = dict_from_json('../catalog.json')
label_encoder.fit(list(catalog.keys()))

#load the model
#load model:
the_model = EmbeddingNet()
the_model.load_state_dict(torch.load('online_model.pth'))

#make the datasets:
plot_dataset = make_plot_dataset(label_encoder, 15)
plot_loader = torch.utils.data.DataLoader(plot_dataset, batch_size=400, shuffle=False)

#extract embeddings and plot:
val_embeddings_tl, val_labels_tl = extract_embeddings(plot_loader, the_model)
plot_embeddings(val_embeddings_tl, val_labels_tl, label_encoder=label_encoder)


