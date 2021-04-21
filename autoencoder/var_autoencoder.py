"""
    https://keras.io/examples/generative/vae/
    https://www.tensorflow.org/guide/keras/custom_layers_and_models
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from contextlib import redirect_stdout
import datetime

from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape
from keras.models import Model
from keras.losses import binary_crossentropy
from data_generator import DataGenerator

tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()

# Seed for reproducability
np.random.seed(25)

# Training parameters
latent_dim = 32  # Dimension of the latent space
batch_size = 2**10  # Batch size for the data generator
epochs = 10  # Epochs to train the model for

logfile_name = "vae_log_" + str(datetime.datetime.now())

def compute_latent(x):
    """
    Samples points from the latent space given a distribution.

    Args:
        x: List of the form [mu, sigma], containing the mean and the
           log-variance of the latent space distribution as Keras tensors.
    Returns:
        Keras tensor containing the sampled point.

    """
    _mu, _sigma = x
    batch = K.shape(_mu)[0]
    dim = K.shape(_mu)[1]
    # generate random noise
    eps = K.random_normal(shape=(batch, dim))
    # exp(sigma/2) converts log-variance to standard deviation
    return _mu + K.exp(_sigma / 2) * eps


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


# A function to display image sequence
def display_image_sequence(start, end, no_of_images):

    new_points = np.linspace(start, end, no_of_images)
    new_images = decoder.predict(new_points)

    # Display some images
    fig, axes = plt.subplots(ncols=no_of_images, sharex="none", sharey="all", figsize=(20, 7))

    for i in range(no_of_images):
        axes[i].imshow(new_images[i])
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.show()


# Input size
img_height = 96
img_width = 96
num_channels = 3
input_shape = (img_height, img_width, num_channels)


# Constructing encoder

# Main encoder block
encoder_input = Input(shape=input_shape)
encoder_conv = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(encoder_input)
encoder_conv = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(encoder_conv)
encoder = Flatten()(encoder_conv)

# Encode mean and variance of latent distribution
mu = Dense(latent_dim)(encoder)
sigma = Dense(latent_dim)(encoder)

# Sampling layer
latent_space = Lambda(compute_latent, output_shape=(latent_dim,))([mu, sigma])

# Save convolution shape to be used in the decoder
conv_shape = K.int_shape(encoder_conv)

# Constructing decoder
decoder_input = Input(shape=(latent_dim,))
decoder = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(decoder_input)
decoder = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(decoder)
decoder_conv = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(decoder)
decoder_conv = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(decoder_conv)
decoder_conv = Conv2DTranspose(filters=num_channels, kernel_size=3, padding='same', activation='sigmoid')(decoder_conv)

# Construct models from the layer blocks
encoder = Model(encoder_input, latent_space, name="Encoder")
decoder = Model(decoder_input, decoder_conv, name="Decoder")
vae = Model(encoder_input, decoder(encoder(encoder_input)), name="Variational Autoencoder")

# Get list of filenames for the data generators
filenames = []
walk = os.walk("../data")
for root, dirs, files in walk:
    for file in files:
        if ".jpg" in file and "model_images" not in root:
            filenames.append(os.path.join(root, file))

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
vae.summary()
encoder.summary()
decoder.summary()

# Train VAE, saving loss history
history = vae.fit(generator, epochs=epochs, validation_data=val_generator)

# Plotting loss value decrease
#plt.plot(history.history['loss'])
#plt.title("Training loss")
#plt.show()
#plt.plot(history.history['val_loss'])
#plt.title("Validation loss")
#plt.show()

# Transform images to points in latent space using encoder
encoded = encoder.predict(generator[0])

# Displaying images in latent space
plt.figure(figsize=(14, 12))
plt.scatter(encoded[:, 0], encoded[:, 1], s=2)
plt.colorbar()
plt.grid()
plt.show()

# Displaying several new images
# Starting point=(0,2), end point=(2,0)
display_image_sequence(np.ones(latent_dim)*(-1), np.ones(latent_dim), 9)


def plot_latent_space(decoder, n=15, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = img_width
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.random.multivariate_normal(np.zeros(latent_dim), np.identity(latent_dim)*3).reshape((1, -1))
            x_decoded = decoder.predict(z_sample)
            digit = np.array(x_decoded).reshape(digit_size, digit_size, 3) * 255
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    figure = figure.astype(int)

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

plot_latent_space(decoder)

pass