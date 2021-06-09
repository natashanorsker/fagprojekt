import os.path
import random

import keras.models
import numpy as np
from random import shuffle
from data_generator import DataGenerator
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

#random.seed(25)
#np.random.seed(25)

model_name = "temp_model"

dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(os.path.join("models", model_name))
encoder = keras.models.load_model("encoder", compile=False)
os.chdir(dir_path)

encoder.summary()

encodings = np.load(os.path.join("models", model_name, "encodings.npy"))
labels = np.load(os.path.join("models", model_name, "labels.npy"), allow_pickle=True)

id = np.random.choice(labels)

target = cv2.imread(id).reshape((1, 96, 96, 3)) / 256

target_enc = encoder(target)


dists = np.sum((encodings - target_enc)**2, axis=1)

closest_ids = np.argsort(dists)[:5]
closest_dists = dists[closest_ids]


fig, axs = plt.subplots(nrows=1, ncols=6)

for ax in axs:
    ax.set_axis_off()

axs[0].set_title("Input image")
axs[0].imshow(target.squeeze())

for i, idx in enumerate(closest_ids):
    axs[i + 1].set_title(str(closest_dists[i]))
    axs[i + 1].imshow(cv2.imread(labels[idx]))
plt.show()
