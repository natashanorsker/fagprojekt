"""
    https://keras.io/examples/generative/vae/
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import scipy
import cv2
from tqdm import tqdm
from data_generator import DataGenerator
from keras import backend as K

# DEBUGGING
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.util import nest
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):

        kl_multiplier = 1

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * kl_multiplier
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 8
input_shape = (200, 200, 3)

encoder_inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(32, 3, activation="relu", strides=5, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=4, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z], name="encoder")
encoder.summary()


latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(10 * 10 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((10, 10, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=4, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=5, padding="same")(x)
x = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder_outputs = layers.Reshape((200, 200, 3, 1))(x)
decoder = keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
decoder.summary()


# Define the VAE loss.
def vae_loss(x, x_decoded_mean):
    """Defines the VAE loss functions as a combination of MSE and KL-divergence loss."""
    mse_loss = K.mean(keras.losses.mse(x, x_decoded_mean), axis=(1, 2)) * 200 * 200
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return mse_loss + kl_loss


#(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np .expand_dims(mnist_digits, -1).astype("float32") /255

# N = sum(len(files) for _, _, files in os.walk("data"))

#images = np.zeros((N, 200, 200, 3), dtype="float32")

filenames = []

walk = tqdm(os.walk("data"))
for root, dirs, files in walk:
    for file in files:
        if ".jpg" in file:
            filenames.append(os.path.join(root, file))

N = len(filenames)

images = np.zeros((N // 4, 200, 200, 3, 1))

tq = tqdm(enumerate(filenames[:N // 4]))
for i, file in tq:
    images[i] = cv2.imread(file).reshape((200, 200, 3, 1)) / 255

generator = DataGenerator(filenames[N//8:])
val_generator = DataGenerator(filenames[:N//8])

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

vae.fit(images, epochs=30, batch_size=128)

#vae.fit_generator(generator=generator,
#                  validation_data=val_generator,
#                  #use_multiprocessing=True,
#                  #workers=6,
#                  epochs=5)


import matplotlib.pyplot as plt


def plot_latent_space(vae, n=15, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 200
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi, 0, 0, 0, 0, 0, 0]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, 3)
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
    plt.imshow(figure, cmap="Greys_r")
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
