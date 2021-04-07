"""
    https://keras.io/examples/generative/vae/
    https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tqdm import tqdm
from autoencoder.data_generator import DataGenerator

IMAGE_SIZE = 96
LATENT_DIM = 32

from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=LATENT_DIM, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv1 = layers.Conv2D(32, 3, activation="relu", strides=3, padding="same")
        self.conv2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.conv3 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(intermediate_dim, activation="relu")

        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")

        self.sampling = Sampling()


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        #self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        #self.dense_output = layers.Dense(original_dim, activation="sigmoid")

        self.dense = layers.Dense(8 * 8 * 64, activation="relu")
        self.reshape = layers.Reshape((8, 8, 64))
        self.conv_trans1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.conv_trans2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.conv_trans3 = layers.Conv2DTranspose(16, 3, activation="relu", strides=3, padding="same")
        self.conv_trans4 = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")
        self.reshape2 = layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 3, 1))

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_trans1(x)
        x = self.conv_trans2(x)
        x = self.conv_trans3(x)
        x = self.conv_trans4(x)
        return self.reshape2(x)


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

latent_dim = 32
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

# encoder_inputs = keras.Input(shape=input_shape)
# x = layers.Conv2D(32, 3, activation="relu", strides=3, padding="same")(encoder_inputs)
# x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Flatten()(x)
# x = layers.Dense(16, activation="relu")(x)
# z_mean = layers.Dense(latent_dim, name="z_mean")(x)
# z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
# z = Sampling()([z_mean, z_log_var])
# encoder = keras.Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z], name="encoder")
# encoder.summary()


#latent_inputs = keras.Input(shape=(latent_dim,))
#x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
#x = layers.Reshape((8, 8, 64))(x)
#x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
#x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
#x = layers.Conv2DTranspose(16, 3, activation="relu", strides=3, padding="same")(x)
#x = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
#decoder_outputs = layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, 3, 1))(x)
#decoder = keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
#decoder.summary()


#(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np .expand_dims(mnist_digits, -1).astype("float32") /255

# N = sum(len(files) for _, _, files in os.walk("data"))

#images = np.zeros((N, 200, 200, 3), dtype="float32")

filenames = []

walk = tqdm(os.walk("../data"))
for root, dirs, files in walk:
    for file in files:
        if ".jpg" in file:
            filenames.append(os.path.join(root, file))

N = len(filenames)

#images = np.zeros((N // 2, IMAGE_SIZE, IMAGE_SIZE, 3, 1))

#tq = tqdm(enumerate(filenames[:N//2]))
#for i, file in tq:
#    images[i] = cv2.imread(file).reshape((IMAGE_SIZE, IMAGE_SIZE, 3, 1)) / 255

generator = DataGenerator(filenames)
#val_generator = DataGenerator(filenames[:N//8])

#vae = VAE(encoder, decoder)
#vae.compile(optimizer=keras.optimizers.Adam())

#vae.fit(images, epochs=1, batch_size=128)

#vae.fit(generator,
#                  validation_data=val_generator,
#                  #use_multiprocessing=True,
#                  #workers=6,
#                  epochs=5)

vae = VariationalAutoEncoder((IMAGE_SIZE, IMAGE_SIZE, 3), 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy())
vae.fit(generator, epochs=1, batch_size=64)


import matplotlib.pyplot as plt


def plot_latent_space(vae, n=15, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = IMAGE_SIZE
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.random.multivariate_normal(np.zeros(latent_dim), np.identity(latent_dim)*3).reshape((1, -1))
            x_decoded = vae.decoder.call(z_sample)
            digit = np.array(x_decoded).reshape(digit_size, digit_size, 3) * 255
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    plt.show()


plot_latent_space(vae)


#def plot_label_clusters(vae, data, labels):
#    # display a 2D plot of the digit classes in the latent space
#    z_mean, _, _ = vae.encoder.predict(data)
#    plt.figure(figsize=(12, 10))
#    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
#    plt.colorbar()
#    plt.xlabel("z[0]")
#    plt.ylabel("z[1]")
#    plt.show()

#(x_train, y_train), _ = keras.datasets.mnist.load_data()
#x_train = np.expand_dims(x_train, -1).astype("float32") / 255

#plot_label_clusters(vae, x_train, y_train)
