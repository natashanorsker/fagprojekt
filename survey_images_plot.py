import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

for d in os.listdir("survey_images"):

    recs = []

    query = cv2.imread(os.path.join("survey_images", d, "query.jpg"))
    query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

    for root, subfolders, files in os.walk(os.path.join("survey_images", d)):
        if not subfolders:
            dir_recs = []
            for file in files:
                im = cv2.imread(os.path.join(root, file))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                dir_recs.append(im)
            recs.append(dir_recs)

    fig, axs = plt.subplots(nrows=4, ncols=5)

    for i in range(4):
        for j in range(5):
            if not i and j == 2:
                axs[i, j].imshow(query)
                axs[i, j].set_title("Query Image")
            if i == 1 and j == 2:
                axs[i, j].set_title("Recommendations")
            if i:
                axs[i, j].imshow(recs[i - 1][j])
            if i and j == 0:
                axs[i, j].set_ylabel("ABC"[i - 1], rotation=0, fontsize=16, labelpad=20)
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                for key, spine in axs[i, j].spines.items():
                    spine.set_visible(False)
            else:
                axs[i, j].set_axis_off()
    plt.tight_layout()
    line = plt.Line2D([0.06, 0.95], [0.48, 0.48], linestyle="--", transform=fig.transFigure, color="grey")
    fig.add_artist(line)
    line = plt.Line2D([0.06, 0.95], [0.24, 0.24], linestyle="--", transform=fig.transFigure, color="grey")
    fig.add_artist(line)
    plt.show()