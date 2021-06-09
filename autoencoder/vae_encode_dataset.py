import os.path
import keras.models
import numpy as np
from data_generator import DataGenerator, get_train_test_split_paths
from tqdm import tqdm


model_name = "temp_model"

dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(os.path.join("models", model_name))
encoder = keras.models.load_model("encoder", compile=False)
decoder = keras.models.load_model("decoder", compile=False)
os.chdir(dir_path)

encoder.summary()
decoder.summary()

seed = 42069
np.random.seed(seed)

train_set, _ = get_train_test_split_paths()

generator = DataGenerator(train_set, batch_size=2**9, ids=True)

latent_dim = decoder.input.shape[1]
N = len(generator) * generator.batch_size #len(filenames)

encodings = np.zeros((N, latent_dim))
labels = np.zeros(N, dtype=object)

for i, (imgs, lbls) in tqdm(enumerate(generator), total=len(generator)):
    labels[generator.batch_size*i:generator.batch_size*(i+1)] = np.array(lbls)
    encodings[generator.batch_size * i:generator.batch_size * (i + 1)] = encoder(imgs)

np.save(os.path.join("models", model_name, "encodings.npy"), encodings)
np.save(os.path.join("models", model_name, "labels.npy"), labels)
