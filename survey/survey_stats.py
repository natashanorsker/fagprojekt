import numpy as np
import pandas as pd
import requests
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt


def bootstrap_mean_rank_plot(data, N, title, labels=None):

    if not labels:
        labels = [None] * len(data)

    assert len(data) == len(labels)

    plt.figure(figsize=(8, 4))
    plt.style.use("bmh")

    for p, label in zip(data, labels):
        bs = np.zeros((3, N, len(p["score"]) // 3))

        for i in tqdm(range(N)):
            for j, mod in enumerate(["d", "a", "r"]):
                slice = p["score"][p["model"] == mod]
                bs[j, i] = np.random.choice(slice, size=len(slice), replace=True)

        bs_means = np.sort(np.mean(bs, axis=2), axis=1)

        percs = np.percentile(bs_means, (2.5, 97.5), axis=1)

        x = np.arange(3)
        y = np.array([np.mean(p["score"][p["model"] == mod]) for mod in ["d", "a", "r"]])

        if label:
            l = label + ", "
        else:
            l = ""
        plt.errorbar(x, y, yerr=np.abs(percs - y), linestyle="--", capsize=4, label=f"{l}95% CI (Bootstrapped, N={N})")

    plt.xticks(x, ["Deep Ranking", "Autoencoder", "Random"])
    # plt.xlabel("Model")
    plt.ylabel("Mean Rank")
    plt.legend()
    plt.title(title)
    plt.show()

urls = [
    "https://docs.google.com/spreadsheets/d/1akG6y1lX4pi7EigF2Y8LNDxW-nR3Rxdfs9MxW_y4pYM/gviz/tq?tqx=out:csv",
    "https://docs.google.com/spreadsheets/d/1Ywph9EmoDuVRL433TMxsoXtsVFS6Pox_VSNoZ68Az38/gviz/tq?tqx=out:csv"
]

seqs = [
    "dradraradrdaradarddraardadrdardradraadrdradarrdadardardraadrrdadarrdaraddardardarrdardaadr",
    "daradrdarrdaradradardadrradadrrdaardrdaradradadrdraarddradraardradardrdaardraddradrardarad"
]


dicts = []

for i in range(2):

    r = requests.get(urls[i])

    data = r.content

    data = pd.read_csv(BytesIO(data), index_col=0)

    new_names = []

    model_sequence = seqs[i]
    i = 0

    for col in data.columns:
        if col[0] == "Q":
            new_names.append(" ".join([col[1:3].replace(":", ""), model_sequence[i], col[-2]]))
            i += 1
        else:
            new_names.append(col)

    d = {orig: new for orig, new in zip(data.columns, new_names)}
    d["Tidsstempel"] = "time"
    d["Køn // Gender"] = "gender"
    d[r"Tak for dine svar! Beskriv venligst hvordan du vurderede kvaliteten af anbefalingerne\\ Thank you for answering! Please describe how you assessed the quality of the recommendations?"] = "comment"

    data.rename(columns=d, inplace=True)

    new_obs = []

    shape_pref = [41, 46, 31, 49, 73, 23, 26, 78, 71, 3, 62, 2]
    col_pref = [75, 6, 11, 43, 64]

    for i, (row_index, row) in enumerate(data.iterrows()): #iterate over rows
        for column_index, value in row.items():
            if value in [1, 2, 3]:
                num, model, letter = column_index.split(" ")
                wild = int(num) % 2 == 0
                if i in shape_pref:
                    pref = "shape"
                elif i in col_pref:
                    pref = "col"
                else:
                    pref = "none"
                new_obs.append([i * 30 + int(num), data["gender"][i].split(" ")[-1], pref, wild, model, value])

    new_data = pd.DataFrame(new_obs, columns=["person", "gender", "pref", "wild", "model", "score"])
    dicts.append(new_data)

crop, no_crop = dicts

crop.to_csv(r'crop_survey_results.csv', index=False)
no_crop.to_csv(r'no_crop_survey_results.csv', index=False)

N = 1000

# All images
p = crop[["model", "score"]]

bootstrap_mean_rank_plot([p], N, "Mean Ranks on all survey images")

# Wild images
p = [crop[["model", "score"]][crop["wild"]], no_crop[["model", "score"]][no_crop["wild"]]]

bootstrap_mean_rank_plot(p, N, "Mean Ranks on wild images", labels=["Segmented", "Not segmented"])

# Catalog images
p = crop[["model", "score"]][np.logical_not(crop["wild"])]

bootstrap_mean_rank_plot([p], N, "Mean Ranks on catalog images")
