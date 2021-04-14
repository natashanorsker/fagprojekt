import torchvision.utils

from nets import EmbeddingNet
from dataset import make_plot_dataset
from sklearn import preprocessing
from utilities import dict_from_json
from utilities import info_from_id
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

writer = SummaryWriter("runs/pandora")

cuda = torch.cuda.is_available()


def plot_embeddings(embeddings, targets, encoder, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(set(targets))))
    markers = {'Bracelets': '*', 'Charms': '.', 'Jewellery spare parts': 'x', 'Necklaces & Pendants': '3', 'Rings': 's',
               'Earrings': 'D', 'Misc': 'p', 'Set': '+'}
    col_num = 0
    targets = encoder.inverse_transform(targets)
    for i in set(targets):
        category = info_from_id(i, master_file_path='../data_code/masterdata.csv')
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[col_num],
                    marker=markers[category])
        col_num += 1
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.legend(list(set(targets)), loc='best')
    plt.legend(list(set(targets)), loc='best')
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
    plot_dataset = make_plot_dataset(label_encoder, 15)
    plot_loader = torch.utils.data.DataLoader(plot_dataset, batch_size=400, shuffle=False)



    #make image grid for tensorboard:
    examples = iter(plot_loader)
    example_data, example_targets = examples.next()

    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('pandora_images', img_grid)
    writer.close()



    # extract embeddings and plot:
    val_embeddings_tl, val_labels_tl = extract_embeddings(plot_loader, the_model)
    plot_embeddings(val_embeddings_tl, val_labels_tl, encoder=label_encoder)
