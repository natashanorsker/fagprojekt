import numpy as np
import matplotlib.pyplot as plt
import os

dir = 'map_npz/labels_and_metals'
models = os.listdir('models')
npz_files = os.listdir(dir)

K = 20
cmcs = np.zeros((len(models), K))
rss = np.zeros((len(models), K))

for i, file in enumerate(npz_files):
    file = dir + file
    f = np.load(file)
    cmcs[i, :] = f['cmcs']
    rss[i, :] = f['rss']

# CMC Curve: Plots TPIR against ranks
models_crop = [mod[7:] for mod in models if 'vae' not in mod]
models_crop2 = [mod[:mod.find('_')] for mod in models_crop]
margins = [mod[mod.find('_',15)+1:][:mod[mod.find('_',15)+1:].find('_')] for mod in models_crop]
leg = [models_crop2[i] + ' '+ margins[i] for i in range(len(margins))]
leg.append('vae')
plt.figure()
plt.plot(np.tile(np.arange(1,K+1), (cmcs.shape[0],1)).T,cmcs.T)
plt.xticks(range(1,K+1))
plt.xlabel('Rank')
plt.ylabel('Identification Accuracy')
plt.title('CMC Curve')
plt.legend(leg)
plt.ylim(0,1.02)
plt.savefig('../Figures/categorycmcc.pdf')
plt.show()

plt.figure()
plt.plot(np.tile(np.arange(1,K+1), (rss.shape[0],1)).T, rss.T)
plt.xticks(range(1,K+1))
plt.xlabel('Rank')
plt.ylabel('Recall')
plt.title('Recall at K')
plt.legend(leg)
plt.ylim(0,1.02)
plt.savefig('../Figures/rekallatKcategories.pdf')
plt.show()
