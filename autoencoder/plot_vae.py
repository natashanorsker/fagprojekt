import os.path

import keras.models
import numpy as np
import matplotlib.pyplot as plt
from data_generator import DataGenerator, get_train_test_split_paths
from sklearn.decomposition import PCA

model_name = "final_model"

dir_path = os.path.dirname(os.path.realpath(__file__))
model_paths = os.listdir("models")
model_path = os.path.join(model_name)

os.chdir(os.path.join("models", model_path))
encoder = keras.models.load_model("encoder", compile=False)
decoder = keras.models.load_model("decoder", compile=False)
os.chdir(dir_path)

encoder.summary()
decoder.summary()

latent_dim = decoder.input.shape[1]

_, filenames = get_train_test_split_paths()

generator = DataGenerator(filenames, batch_size=2**9, labels=True)


def plot_latent_space(decoder, n=9, figsize=15):
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
    plt.xticks([])
    plt.yticks([])
    plt.imshow(figure)
    plt.title(f"{n}x{n} " + r"images sampled from the latent space $\sim \mathcal{N}(\mathbf{0},\mathbf{I})$")
    plt.show()


# A function to display image sequence
def display_image_sequence(start, end, no_of_images):

    new_points = np.linspace(start, end, no_of_images)
    new_images = decoder.predict(new_points)

    # Display some images
    fig, axes = plt.subplots(ncols=no_of_images, sharex="none", sharey="all", figsize=(10, 4))

    for i in range(no_of_images):
        axes[i].imshow(new_images[i])
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    plt.show()


# Transform images to points in latent space using encoder
imgs, labels = generator[0]
encoded = encoder.predict(imgs)

pca = PCA(n_components=2)
pca.fit(encoded.T)


d = {v: i for i, v in enumerate(list(set(labels)))}

cols = [d[l] for l in labels]

# Displaying images in latent space
plt.figure(figsize=(14, 12))
plt.scatter(pca.components_[0], pca.components_[1], s=10, c=cols)
plt.colorbar()
plt.grid()
plt.show()


plot_latent_space(decoder)

display_image_sequence(-np.ones(latent_dim), np.ones(latent_dim), 9)
