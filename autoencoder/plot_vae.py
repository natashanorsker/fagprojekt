import os.path

import keras.models
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from data_generator import DataGenerator
from utilities import info_from_id

dir_path = os.path.dirname(os.path.realpath(__file__))
model_paths = os.listdir("models")
model_path = os.path.join(model_paths[-1])

os.chdir(os.path.join("models", model_path))
encoder = keras.models.load_model("encoder", compile=False)
decoder = keras.models.load_model("decoder", compile=False)
os.chdir(dir_path)

encoder.summary()
decoder.summary()

latent_dim = decoder.input.shape[1]

# Get list of filenames for the data generators
filenames = []
walk = os.walk("../data")
for root, dirs, files in walk:
    for file in files:
        if ".jpg" in file and "model_images" not in root:
            filenames.append(os.path.join(root, file))

# Less files when debugging
#filenames = filenames[:len(filenames)//5]

# Constructing data generators
shuffle(filenames)
generator = DataGenerator(filenames, batch_size=128, labels=True)

def plot_latent_space(decoder, n=15, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 96
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


# Transform images to points in latent space using encoder
imgs, labels = generator[0]
encoded = encoder.predict(imgs)

# Displaying images in latent space
plt.figure(figsize=(14, 12))
plt.scatter(encoded[:, 0], encoded[:, 1], s=2, c=labels)
plt.colorbar()
plt.grid()
plt.show()


plot_latent_space(decoder)

display_image_sequence(-np.ones(latent_dim), np.ones(latent_dim), 9)
