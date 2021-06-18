import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
import os
import cv2
from tqdm import tqdm


# the three first letters on the code on the plot signifies the ordering, eg.
# adrradaad means autoencoder, deep ranking, then random

for d in tqdm(os.listdir("survey_images")):

    recs = []

    query = cv2.imread(os.path.join("survey_images", d, f"{d}_query.jpg"))
    query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

    for root, subfolders, files in os.walk(os.path.join("survey_images", d)):

        if not subfolders:
            dir_recs = []
            for file in files:
                im = cv2.imread(os.path.join(root, file))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                dir_recs.append(im)
            recs.append(dir_recs)

    ids = np.random.choice([0, 1, 2], replace=False, size=3)

    hr = np.ones((4, 5))

    gridspec_kw = {"height_ratios": [3, 1, 1, 1]}

    fig, axs = plt.subplots(nrows=4, ncols=5, gridspec_kw=gridspec_kw)

    for ax in axs[0, 1:4]:
        ax.remove()

    gs = GridSpec(4, 5, **gridspec_kw)
    ax = fig.add_subplot(gs[0, 1:4])

    ax.imshow(query)
    ax.set_title("Query Image")
    ax.set_axis_off()

    for i in range(4):
        for j in range(5):

            if i or j not in [1, 2, 3]:
                ax = axs[i, j]

                if not i and j == 4:
                    ax.text(
                        0.5, 0.5, "".join(["dar"[i] for i in ids]) + "".join(np.random.choice(list("dar"), size=6)))
                if i == 1 and j == 2:
                    ax.set_title("Recommendations")
                if i:
                    ax.imshow(recs[ids[i - 1]][j])
                if i and j == 0:
                    ax.set_ylabel("ABC"[i - 1], rotation=0, fontsize=16, labelpad=20)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for key, spine in ax.spines.items():
                        spine.set_visible(False)
                else:
                    ax.set_axis_off()
    plt.tight_layout()
    line = plt.Line2D([0.06, 0.95], [0.37, 0.37], linestyle="--", transform=fig.transFigure, color="grey")
    fig.add_artist(line)
    line = plt.Line2D([0.06, 0.95], [0.18, 0.18], linestyle="--", transform=fig.transFigure, color="grey")
    fig.add_artist(line)
    plt.savefig(os.path.join("survey_images", d, f"{d}_questionnaire.jpg"))
    plt.close()