"""
    https://keras.io/examples/generative/vae/
    https://www.tensorflow.org/guide/keras/custom_layers_and_models
    https://becominghuman.ai/using-variational-autoencoder-vae-to-generate-new-images-14328877e88d
"""

import os
import numpy as np
import tensorflow as tf
from random import shuffle
from contextlib import redirect_stdout
import datetime

tf.compat.v1.disable_eager_execution()
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape
from keras.models import Model
from keras.losses import binary_crossentropy
from data_generator import DataGenerator, get_train_test_split_paths
tf.executing_eagerly()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Seed for reproducability
seed = 42069
np.random.seed(seed)

train_set, _ = get_train_test_split_paths()


def train_vae(epochs=10, latent_dim=64, batch_size=2**9,
              logging=True, save_model=True, model_name=None):
    """


    Args:


    Returns:
        None
    """

    start_time = datetime.datetime.now()

    if logging:
        logfile_name = f"vae_log_{start_time}.txt"

        with open("../logs/vae/"+logfile_name, 'a+') as f:
            with redirect_stdout(f):
                print(f"VAE run at {start_time}\n")
                print(f"Random seed: {seed}")
                print(f"Epochs: {epochs}")
                print(f"Batch size: {batch_size}")
                print(f"Latent dimension: {latent_dim}\n\n\n")

    # Input size
    img_height = 96
    img_width = 96
    num_channels = 3
    input_shape = (img_height, img_width, num_channels)

    # Constructing encoder

    # Main encoder block
    encoder_input = Input(shape=input_shape, name="encoder_input")
    encoder_conv1 = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu', name="convolution_1")(encoder_input)
    encoder_conv2 = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu', name="convolution_2")(encoder_conv1)
    encoder_conv3 = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name="convolution_3")(encoder_conv2)
    encoder = Flatten(name="flatten")(encoder_conv3)

    # Encode mean and variance of latent distribution
    mu = Dense(latent_dim, name="mu")(encoder)
    sigma = Dense(latent_dim, name="sigma")(encoder)

    def compute_latent(x):
        """
        Samples points from the latent space given a distribution.

        Args:
            x: List of the form [mu, sigma], containing the mean and the
               log-variance of the latent space distribution as Keras tensors.
        Returns:
            Keras tensor representing the sampled point.

        """
        mu, sigma = x
        batch = K.shape(mu)[0]
        dim = K.shape(mu)[1]
        # generate random noise
        eps = K.random_normal(shape=(batch, dim))
        # exp(sigma/2) converts log-variance to standard deviation
        return mu + K.exp(sigma / 2) * eps

    # Sampling layer
    latent_space = Lambda(compute_latent, output_shape=(latent_dim,), name="latent_space")([mu, sigma])

    # Save convolution shape to be used in the decoder
    conv_shape = K.int_shape(encoder_conv3)

    # Constructing decoder
    decoder_input = Input(shape=(latent_dim,), name="decoder_input")
    decoder_dense = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu', name="decoder_dense")(decoder_input)
    decoder_reshape = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]), name="decoder_reshape")(decoder_dense)
    decoder_conv1 = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu', name="transposed_convolution_1")(decoder_reshape)
    decoder_conv2 = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu', name="transposed_convolution_2")(decoder_conv1)
    decoder_conv3 = Conv2DTranspose(filters=num_channels, kernel_size=3, strides=2, padding='same', activation='sigmoid', name="transposed_convolution_3")(decoder_conv2)

    # Construct models from the layer blocks
    encoder = Model(encoder_input, latent_space, name="Encoder")
    decoder = Model(decoder_input, decoder_conv3, name="Decoder")
    vae = Model(encoder_input, decoder(encoder(encoder_input)), name="VAE")

    s = int(len(train_set) // 10)
    train_data = train_set[s:]
    val_data = train_set[:s]

    # Constructing data generators
    generator = DataGenerator(train_data, batch_size=batch_size)
    val_generator = DataGenerator(val_data, batch_size=batch_size)

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

    # Compile the model
    vae.compile(optimizer='adam', loss=kl_reconstruction_loss)

    # Print model info
    if logging:
        with open("../logs/vae/"+logfile_name, 'a+') as f:
            with redirect_stdout(f):
                vae.summary()
                encoder.summary()
                decoder.summary()

    # Train VAE, saving loss history
    history = vae.fit(generator, epochs=epochs, validation_data=val_generator)

    if logging:
        with open("../logs/vae/"+logfile_name, 'a+') as f:
            with redirect_stdout(f):
                print("\n\n")
                print("Loss history:")
                print(history)

    if save_model:
        # Save model
        model_name = "temp_model" if model_name is None else f"{model_name}_vae_{start_time}"
        tf.keras.models.save_model(encoder, os.path.join("models", model_name, "encoder"))
        tf.keras.models.save_model(decoder, os.path.join("models", model_name, "decoder"))


if __name__ == "__main__":
    train_vae(epochs=10, model_name="temp_model", latent_dim=64)
