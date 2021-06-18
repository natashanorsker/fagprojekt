import numpy as np
import pandas as pd
import requests
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt


def bootstrap_mean_rank_plot(p, N, title):

    bs = np.zeros((3, N, len(p["score"]) // 3))

    for i in tqdm(range(N)):
        for j, mod in enumerate(["d", "a", "r"]):
            slice = p["score"][p["model"] == mod]
            bs[j, i] = np.random.choice(slice, size=len(slice), replace=True)

    bs_means = np.sort(np.mean(bs, axis=2), axis=1)

    percs = np.percentile(bs_means, (2.5, 97.5), axis=1)

    x = np.arange(3)
    y = np.array([np.mean(p["score"][p["model"] == mod]) for mod in ["d", "a", "r"]])

    plt.figure(figsize=(8, 4))
    plt.style.use("bmh")
    plt.errorbar(x, y, yerr=np.abs(percs - y), linestyle="--", capsize=4, label=f"95% CI (Bootstrapped, N={N}) ")
    plt.xticks(x, ["Deep Ranking", "Autoencoder", "Random"])
    # plt.xlabel("Model")
    plt.ylabel("Mean Rank")
    plt.legend()
    plt.title(title)
    plt.show()


r = requests.get("https://docs.google.com/spreadsheets/d/1Ywph9EmoDuVRL433TMxsoXtsVFS6Pox_VSNoZ68Az38/gviz/tq?tqx=out:csv")



data = r.content

data = pd.read_csv(BytesIO(data), index_col=0)

new_names = []

model_sequence = "daradrdarrdaradradardadrradadrrdaardrdaradradadrdraarddradraardradardrdaardraddradrardarad"
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

print(data.head)

new_obs = []

shape_pref = [41, 46, 31, 49, 73, 23, 26, 78, 71, 3, 62, 2]
col_pref = [75, 6, 11, 43, 64]

for i, (row_index, row) in enumerate(data.iterrows()): #iterate over rows
    if i == 2:
        print(row[10])
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

new_data.to_csv(r'survey_results.csv', index = False)


N = 1000

# All images
p = new_data[["model", "score"]]

bootstrap_mean_rank_plot(p, N, "Mean Ranks on all survey images")

# Wild images
p = new_data[["model", "score"]][new_data["wild"]]

bootstrap_mean_rank_plot(p, N, "Mean Ranks on wild images")

# Catalog images
p = new_data[["model", "score"]][np.logical_not(new_data["wild"])]

bootstrap_mean_rank_plot(p, N, "Mean Ranks on catalog images")

# color preference (not used in report)
p = new_data[["model", "score"]][new_data["pref"] == "col"]

bootstrap_mean_rank_plot(p, N, "Mean Ranks for people who prefer color")

# shape preference (not used in report)
p = new_data[["model", "score"]][new_data["pref"] == "shape"]

bootstrap_mean_rank_plot(p, N, "Mean Ranks for people who prefer shape")
