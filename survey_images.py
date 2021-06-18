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

# fill folder if empty
for d in os.listdir("Questionnaire_imgs"):
    dd = os.path.join("survey_images", d[:-4])
    if not os.path.isdir(dd):
        os.mkdir(dd)
        os.mkdir(os.path.join(dd, "autoencoder"))
        os.mkdir(os.path.join(dd, "deepranking"))
        os.mkdir(os.path.join(dd, "random"))
        im = cv2.imread(os.path.join("Questionnaire_imgs", d))
        cv2.imwrite(os.path.join(dd, "query.jpg"), im)


#ugly hack
def extract_embeddings(dataloader, model, force_no_cuda=False):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 10))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda and not force_no_cuda:
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

# load pytorch tensors and model
# TODO: in the future the tensors should just be loaded from a file
model = EmbeddingNet()
model.load_state_dict(torch.load('deepRanking/models/online_semi_model_margin_0.1_2021-06-13_0.0487loss.pth',
                                 map_location=torch.device('cpu')))
model.eval()

catalog = dict_from_json('catalog.json')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(catalog.keys()))

#make the 'normal' datasets:
train_dataset, test_dataset = make_dataset(label_encoder, n_val_products=100, NoDuplicates=False)

#make the dataloaders:
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=False)

embeddingsP, labelsP = extract_embeddings(data_loader, model, force_no_cuda=True)

np.save("embeddings.npy", embeddingsP)
np.save("labels.npy", labelsP)

for d in os.listdir("survey_images"):
    print(d)

    query_img = cv2.imread(os.path.join("survey_images", d, "query.jpg"))

    if "wild" in d:
        try:
            query_img = detectron2segment.inference.extractjewel(query_img)
            cv2.imwrite(os.path.join("survey_images", d, "cropped.jpg"), query_img)
        except Exception as e:
            print(e)
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

    sort_idx = np.argsort(dists)

    sorted_labels = vae_labels[sort_idx]

    sorted_labels_cut = np.array([label[:-10] for label in sorted_labels])

    _, unique = np.unique(sorted_labels_cut, return_index=True)

    recs = sorted_labels[np.sort(unique)][:5]


    for i, rec in enumerate(recs):
        p = "/".join(rec.split(os.sep)[1:])
        p = p[:-10] + f"_00_OG.jpg"
        im = cv2.imread(p)
        p = os.path.join("survey_images", d, "autoencoder", f"{i}_" + p.split(os.sep)[-1])
        cv2.imwrite(p, im)

    # Triplet
    crop_img = cv2.cvtColor((query_img * 255).astype("uint8"), cv2.COLOR_BGR2RGB)
    query_img = Image.fromarray(crop_img)

    transform = transforms.Compose([transforms.Resize((96,96)),
                                 transforms.ToTensor()])
    img_t = transform(query_img)
    batch_t = torch.unsqueeze(img_t, 0)

    # Generate prediction
    with torch.no_grad():
        embedding = model(batch_t)

    dists = np.sum((embeddingsP - embedding.numpy()) ** 2, axis=1)

    sort_idx = np.argsort(dists)

    sorted_labels = np.array(train_dataset.list_IDs)[sort_idx]

    sorted_labels_cut = np.array([label[:-10] for label in sorted_labels])

    _, unique = np.unique(sorted_labels_cut, return_index=True)

    recs = sorted_labels[np.sort(unique)][:5]

    for i, rec in enumerate(recs):
        p = "/".join(rec.split(os.sep))
        p = p[:-10] + f"_00_OG.jpg"
        im = cv2.imread(p)

        p = os.path.join("survey_images", d, "deepranking", f"{i}_" + p.split(os.sep)[-1])

        cv2.imwrite(p, im)
