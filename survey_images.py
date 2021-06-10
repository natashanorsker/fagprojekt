import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy.random
import detectron2segment.inference
from autoencoder.data_generator import get_train_test_split_paths
import keras

numpy.random.seed(42069)

model_name = "final_model"

dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(os.path.join("autoencoder", "models", model_name))
encoder = keras.models.load_model("encoder", compile=False)
os.chdir(dir_path)

encoder.summary()

encodings = np.load(os.path.join("autoencoder", "models", model_name, "encodings.npy"))
labels = np.load(os.path.join("autoencoder", "models", model_name, "labels.npy"), allow_pickle=True)

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
    recs = numpy.random.choice(paths, size=5)

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



    # TODO: Deep ranking recommendations

