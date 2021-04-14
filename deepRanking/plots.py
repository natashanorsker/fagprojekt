from deep_ranking_utils import plot_embeddings, EmbeddingNet, TripletNet, extract_embeddings
from dataset import make_dataset, make_plot_dataset
import torch

#load the model
#load model:
the_model = EmbeddingNet()
the_model.load_state_dict(torch.load('online_model.pth'))

#make the datasets:
plot_dataset = make_plot_dataset(10)
plot_loader = torch.utils.data.DataLoader(plot_dataset, batch_size=20, shuffle=False)

#make the 'normal' datasets:
train_dataset, test_dataset, label_encoder = make_dataset()
#make the dataloaders:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)

val_embeddings_tl, val_labels_tl = extract_embeddings(plot_loader, the_model)
#plot_embeddings(val_embeddings_tl, val_labels_tl)

#
#val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, the_model)
#plot_embeddings(val_embeddings_tl, val_labels_tl)


