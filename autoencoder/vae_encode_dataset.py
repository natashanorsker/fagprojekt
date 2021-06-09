import os.path
import random

import keras.models
import numpy as np
from random import shuffle
from data_generator import DataGenerator
from tqdm import tqdm

random.seed(25)
np.random.seed(25)

model_name = "temp_model"

dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(os.path.join("models", model_name))
encoder = keras.models.load_model("encoder", compile=False)
decoder = keras.models.load_model("decoder", compile=False)
os.chdir(dir_path)

encoder.summary()
decoder.summary()

# Get list of filenames for the data generators
filenames = []
walk = os.walk("../data")
for root, dirs, files in walk:
    for file in files:
        if ".jpg" in file and "model_images" not in root and "not_in_master" not in root:
            filenames.append(os.path.join(root, file))



# Constructing data generators
shuffle(filenames)
generator = DataGenerator(filenames, batch_size=2**9, ids=True)

latent_dim = decoder.input.shape[1]
N = len(generator) * generator.batch_size #len(filenames)

encodings = np.zeros((N, latent_dim))
labels = np.zeros(N, dtype=object)

for i, (imgs, lbls) in tqdm(enumerate(generator), total=len(generator)):
    labels[generator.batch_size*i:generator.batch_size*(i+1)] = np.array(lbls)
    encodings[generator.batch_size * i:generator.batch_size * (i + 1)] = encoder(imgs)

np.save(os.path.join("models", model_name, "encodings.npy"), encodings)
np.save(os.path.join("models", model_name, "labels.npy"), labels)
