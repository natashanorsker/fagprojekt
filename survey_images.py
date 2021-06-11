import os

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn import preprocessing
from torchvision import transforms

import detectron2segment.inference
from autoencoder.data_generator import get_train_test_split_paths
from deepRanking.dataset import make_dataset
from deepRanking.nets import EmbeddingNet
from deepRanking.plots import extract_embeddings
from utilities import dict_from_json

np.random.seed(42069)

model_name = "final_model"

dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(os.path.join("autoencoder", "models", model_name))
encoder = keras.models.load_model("encoder", compile=False)
os.chdir(dir_path)

encoder.summary()

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

    query_img = cv2.imread(os.path.join("survey_images", d, "query.jpg")) / 256

    try:
        query_img = detectron2segment.inference.extractjewel(query_img)
    except Exception:
        query_img = query_img

    query_img = cv2.resize(query_img, (96, 96))

    # random images
    paths, _ = get_train_test_split_paths(test_proportion=0)
    paths = [p for p in paths if "_00_OG" in p]
    recs = np.random.choice(paths, size=5)

    for rec in recs:
        im = cv2.imread(rec)
        p = os.path.join("survey_images", d, "random", os.path.split(rec)[-1])
        cv2.imwrite(p, im)

    # Autoencoder
    target_enc = encoder(query_img.reshape((1, 96, 96, 3)))

    dists = np.sum((encodings - target_enc) ** 2, axis=1)

    closest_ids = np.argsort(dists)[:5]

    recs = labels[closest_ids]

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

