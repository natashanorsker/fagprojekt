import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import numpy.random
from deepRanking.nets import EmbeddingNet
import torch
import detectron2segment.inference
from autoencoder.data_generator import get_train_test_split_paths
import keras
from deepRanking.dataset import make_dataset
import detectron2segment.inference
from utilities import dict_from_json
from sklearn import preprocessing
from torchvision import transforms
from PIL import Image

cuda = torch.cuda.is_available()


# ugly hack
def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 10))
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



# random setup
numpy.random.seed(42069)
paths, _ = get_train_test_split_paths()
paths = [p for p in paths if "_00_OG" in p]

# autoencoder setup
model_name = "final_model"

dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(os.path.join("autoencoder", "models", model_name))
encoder = keras.models.load_model("encoder", compile=False)
os.chdir(dir_path)

vae_embeddings = np.load(os.path.join("autoencoder", "models", model_name, "encodings.npy"))
vae_labels = np.load(os.path.join("autoencoder", "models", model_name, "labels.npy"), allow_pickle=True)

# deep ranking setup
model = EmbeddingNet()
model.load_state_dict(torch.load('deepRanking/models/online_model7-6_0.3979loss.pth', map_location=torch.device('cpu')))

encodings = np.load(os.path.join("autoencoder", "models", model_name, "encodings.npy"))
labels = np.load(os.path.join("autoencoder", "models", model_name, "labels.npy"), allow_pickle=True)

# load pytorch tensors and model
# TODO: in the future the tensors should just be loaded from a file
model = EmbeddingNet()
model.load_state_dict(torch.load('deepRanking/models/online_model7-6_0.3979loss.pth', map_location=torch.device('cpu')))
model.eval()

catalog = dict_from_json('catalog.json')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(catalog.keys()))

#make the 'normal' datasets:
train_dataset, test_dataset = make_dataset(label_encoder, n_test_products=100, NoDuplicates=False)

dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
#make the dataloaders:
data_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)

embeddingsP, labelsP = extract_embeddings(data_loader, model)

for d in os.listdir("survey_images"):
    print(d)

    query_img = cv2.imread(os.path.join("survey_images", d, "query.jpg"))

    try:
        query_img = detectron2segment.inference.extractjewel(query_img)
    except Exception:
        query_img = query_img

    query_img = cv2.resize(query_img, (96, 96)) / 256

    # random images
    recs = numpy.random.choice(paths, size=5)

    for rec in recs:
        im = cv2.imread(rec)
        p = os.path.join("survey_images", d, "random", os.path.split(rec)[-1])
        cv2.imwrite(p, im)

    # Autoencoder
    target_enc = encoder(query_img.reshape((1, 96, 96, 3)))

    dists = np.sum((vae_embeddings - target_enc) ** 2, axis=1)

    closest_ids = np.argsort(dists)[:5]

    recs = vae_labels[closest_ids]

    for rec in recs:
        p = "/".join(rec.split(os.sep)[1:])
        p = p[:-10] + "_00_OG.jpg"
        im = cv2.imread(p)
        p = os.path.join("survey_images", d, "autoencoder", p.split(os.sep)[-1])
        cv2.imwrite(p, im)

    # Triplet
    crop_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    query_img = Image.fromarray(crop_img)

    transform = transforms.Compose([transforms.Resize((96,96)),
                                 transforms.ToTensor()])
    img_t = transform(query_img)
    batch_t = torch.unsqueeze(img_t, 0)

    # Generate prediction
    with torch.no_grad():
        embedding = model(batch_t)

    dists = np.sum((embeddingsP - embedding) ** 2, axis=1)

    closest_ids = np.argsort(dists)[:5]

    recs = dataset[closest_ids]  #this is not possible # .numpy() maybe needed

    for rec in recs:
        p = "/".join(rec.split(os.sep)[1:])
        p = p[:-10] + "_00_OG.jpg"
        im = cv2.imread(p)
        p = os.path.join("survey_images", d, "autoencoder", p.split(os.sep)[-1])
        cv2.imwrite(p, im)

    # TODO: Deep ranking recommendations
    model = EmbeddingNet()
    model.load_state_dict(
        torch.load('deepRanking/models/online_model7-6_0.3979loss.pth', map_location=torch.device('cpu')))

