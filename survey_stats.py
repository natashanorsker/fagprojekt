import numpy as np
import pandas as pd
import requests
from io import BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt


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

for i, (row_index, row) in enumerate(data.iterrows()): #iterate over rows
    for column_index, value in row.items():
        if value in [1, 2, 3]:
            num, model, letter = column_index.split(" ")
            wild = int(num) % 2 == 0
            new_obs.append([i * 30 + int(num), data["gender"][i].split(" ")[-1], wild, model, value])

new_data = pd.DataFrame(new_obs, columns=["person", "gender", "wild", "model", "score"])

new_data.to_csv(r'survey_results.csv', index = False)

p = new_data[["model", "score"]]

N = 10000

bs = np.zeros((3, N, len(p["score"]) // 3))

for i in tqdm(range(N)):
    for j, mod in enumerate(["a", "d", "r"]):
        slice = p["score"][p["model"] == mod]
        bs[j, i] = np.random.choice(slice, size=len(slice), replace=True)

bs_means = np.sort(np.mean(bs, axis=2), axis=1)

print(bs_means)
print(np.percentile(bs_means, (2.5, 97.5), axis=1))