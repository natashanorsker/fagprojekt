"""
    https://keras.io/examples/generative/vae/
    https://www.tensorflow.org/guide/keras/custom_layers_and_models
    https://becominghuman.ai/using-variational-autoencoder-vae-to-generate-new-images-14328877e88d
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from contextlib import redirect_stdout
import datetime

tf.compat.v1.disable_eager_execution()
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape
from keras.models import Model
from keras.losses import binary_crossentropy
from data_generator import DataGenerator
tf.executing_eagerly()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Seed for reproducability
seed = 25
np.random.seed(seed)

# Training parameters
latent_dim = 32  # Dimension of the latent space
batch_size = 2**9  # Batch size for the data generator
epochs = 1  # Epochs to train the model for

start_time = datetime.datetime.now()
logfile_name = f"vae_log_{start_time}.txt"

with open("../logs/vae/"+logfile_name, 'a+') as f:
    with redirect_stdout(f):
        print(f"VAE run at {start_time}\n")
        print(f"Random seed: {seed}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Latent dimension: {latent_dim}\n\n\n")


def compute_latent(x):
    """
    Samples points from the latent space given a distribution.

    Args:
        x: List of the form [mu, sigma], containing the mean and the
           log-variance of the latent space distribution as Keras tensors.
    Returns:
        Keras tensor containing the sampled point.

    """
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.shape(mu)[1]
    # generate random noise
    eps = K.random_normal(shape=(batch, dim))
    # exp(sigma/2) converts log-variance to standard deviation
    return mu + K.exp(sigma / 2) * eps


def kl_reconstruction_loss(true, predicted):
    """
        Samples points from the latent space given a distribution.

        Args:
            true: Input to the model. 96x96x3 image.
            predicted: Output of the model. 96x96x3 image.
        Returns:
            Loss computed as the mean of the reconstruction loss and the Kullback-Leibner loss.

    """
    # Reconstruction loss (binary crossentropy)
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(predicted)) * img_width * img_height

    # KL divergence loss
    kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    # Total loss = 50% rec + 50% KL divergence loss
    return K.mean(reconstruction_loss + kl_loss)


# Input size
img_height = 96
img_width = 96
num_channels = 3
input_shape = (img_height, img_width, num_channels)


# Constructing encoder

# Main encoder block
encoder_input = Input(shape=input_shape)
encoder_conv1 = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(encoder_input)
encoder_conv2 = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(encoder_conv1)
encoder = Flatten()(encoder_conv2)

# Encode mean and variance of latent distribution
mu = Dense(latent_dim)(encoder)
sigma = Dense(latent_dim)(encoder)

# Sampling layer
latent_space = Lambda(compute_latent, output_shape=(latent_dim,))([mu, sigma])

# Save convolution shape to be used in the decoder
conv_shape = K.int_shape(encoder_conv2)

# Constructing decoder
decoder_input = Input(shape=(latent_dim,))
decoder_dense = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(decoder_input)
decoder_reshape = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(decoder_dense)
decoder_conv1 = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(decoder_reshape)
decoder_conv2 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(decoder_conv1)
decoder_conv3 = Conv2DTranspose(filters=num_channels, kernel_size=3, padding='same', activation='sigmoid')(decoder_conv2)

# Construct models from the layer blocks
encoder = Model(encoder_input, latent_space, name="Encoder")
decoder = Model(decoder_input, decoder_conv3, name="Decoder")
vae = Model(encoder_input, decoder(encoder(encoder_input)), name="VAE")

# Get list of filenames for the data generators
filenames = []
walk = os.walk("../data")
for root, dirs, files in walk:
    for file in files:
        if ".jpg" in file and "model_images" not in root:
            filenames.append(os.path.join(root, file))

# Less files when debugging
#filenames = filenames[:len(filenames)//5]

N = len(filenames)
idx = (N // 10) * 9
shuffle(filenames)
train_data = filenames[:idx]
val_data = filenames[idx:]

# Constructing data generators
generator = DataGenerator(train_data, batch_size=batch_size)
val_generator = DataGenerator(val_data, batch_size=batch_size)

# Compile the model
vae.compile(optimizer='adam', loss=kl_reconstruction_loss)

# Print model info
with open("../logs/vae/"+logfile_name, 'a+') as f:
    with redirect_stdout(f):
        vae.summary()
        encoder.summary()
        decoder.summary()

# Train VAE, saving loss history
history = vae.fit(generator, epochs=epochs, validation_data=val_generator)

with open("../logs/vae/"+logfile_name, 'a+') as f:
    with redirect_stdout(f):
        print("\n\n")
        print("Loss history:")
        print(history)

# Save model
model_name = "temp_model"  # f"vae_{start_time}"
tf.keras.models.save_model(encoder, os.path.join("models", model_name, "encoder"))
tf.keras.models.save_model(decoder, os.path.join("models", model_name, "decoder"))


# Plotting loss value decrease
#plt.plot(history.history['loss'])
#plt.title("Training loss")
#plt.show()
#plt.plot(history.history['val_loss'])
#plt.title("Validation loss")
#plt.show()

