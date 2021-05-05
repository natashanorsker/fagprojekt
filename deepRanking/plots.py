import torchvision.utils

from nets import EmbeddingNet
from dataset import make_dataset
from sklearn import preprocessing
from utilities import dict_from_json
from utilities import labels_from_ids
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cuda = torch.cuda.is_available()


def plot_embeddings(embeddings, targets, encoder, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(set(targets))))
    markers = {'Bracelets': '*', 'Charms': '.', 'Jewellery spare parts': 'x', 'Necklaces & Pendants': '3', 'Rings': 's',
               'Earrings': 'D', 'Misc': 'p', 'Set': '+'}
    col_num = 0
    targets = encoder.inverse_transform(targets)
    legend = []

    for i in set(targets):
        category = labels_from_ids(i, master_file_path='data_code/masterdata.csv')
        inds = np.where(targets == i)[0]
        if category not in legend:
            legend += [category]
            label = True

        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[col_num],
                    marker=markers[category], label=category if label else '')

        col_num += 1
        label=False

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])


    #plt.legend(list(set(targets)), loc='best')
    #plt.legend(list(markers.keys()), loc='best')
    plt.legend()
    plt.savefig('online_embedding_plot.png')
    plt.show()


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
        labels = labels.astype(int)
    return embeddings, labels


if __name__ == "__main__":
    label_encoder = preprocessing.LabelEncoder()
    catalog = dict_from_json('../catalog.json')
    label_encoder.fit(list(catalog.keys()))

    # load the model
    # load model:
    the_model = EmbeddingNet()
    the_model.load_state_dict(torch.load('online_model.pth'))

    # make the datasets:
    trainset, testset = make_dataset(label_encoder, 15)
    plot_loader = torch.utils.data.DataLoader(testset, batch_size=400, shuffle=False)


    # extract embeddings and plot:
    val_embeddings_tl, val_labels_tl = extract_embeddings(plot_loader, the_model)
    plot_embeddings(val_embeddings_tl, val_labels_tl, encoder=label_encoder)
