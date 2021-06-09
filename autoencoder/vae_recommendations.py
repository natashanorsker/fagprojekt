import os.path
import random

import keras.models
import numpy as np
from data_generator import get_train_test_split_paths
import matplotlib.pyplot as plt
import cv2


seed = 42069
np.random.seed(seed)

model_name = "temp_model"

dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(os.path.join("models", model_name))
encoder = keras.models.load_model("encoder", compile=False)
os.chdir(dir_path)

encoder.summary()

encodings = np.load(os.path.join("models", model_name, "encodings.npy"))
labels = np.load(os.path.join("models", model_name, "labels.npy"), allow_pickle=True)

_, test_set = get_train_test_split_paths()

ids = np.random.choice(test_set, size=10)

targets = []
most_similar = []

for id_ in ids:
    target = cv2.imread(id_).reshape((1, 96, 96, 3)) / 256

    target_enc = encoder(target)

    dists = np.sum((encodings - target_enc)**2, axis=1)

    closest_ids = np.argsort(dists)[:5]

    targets.append(target)
    most_similar.append(closest_ids)


fig, axs = plt.subplots(nrows=5, ncols=12, figsize=(20, 8))


plt.rc('figure', titlesize=14)
fig.suptitle("VAE Recommendations")

for row in axs:
    for ax in row:
        ax.set_axis_off()

for side in range(2):
    for row in range(5):
        for col in range(6):
            if not row:
                axs[row, col + (6*side)].set_title("Query image" if not col else str(col))
            if not col:
                axs[row, col + (6*side)].margins(x=1)
                im = cv2.cvtColor(targets[row + (side * 5)].squeeze().astype(np.float32), cv2.COLOR_RGB2BGR)
                axs[row, col + (6*side)].imshow(im)
            else:
                img_id = most_similar[row + (side * 5)][col - 1]
                im = cv2.cvtColor(cv2.imread(labels[img_id]).reshape((96, 96, 3)).astype(np.float32) / 256, cv2.COLOR_RGB2BGR)
                axs[row, col + (6 * side)].imshow(im)

plt.tight_layout()
plt.show()

