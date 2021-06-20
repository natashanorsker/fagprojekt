import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn')

dir = 'map_npz/subcategories/'
models = os.listdir('models')
npz_files = os.listdir(dir)

K = 20
cmcs = np.zeros((len(models)+2, K))
rss = np.zeros((len(models)+2, K))

for i, file in enumerate(npz_files):
    file = dir + file
    f = np.load(file)
    cmcs[i, :] = f['cmcs']
    rss[i, :] = f['rss']

# CMC Curve: Plots TPIR against ranks

leg = [model.split('_')[1] + ' ' + model.split('_')[4] for model in models]
leg.append('random')
leg.append('vae')

plt.figure(figsize=(6,4))
plt.plot(np.tile(np.arange(1,K+1), (cmcs.shape[0],1)).T,cmcs.T)
plt.xticks(range(1,K+1))
plt.xlabel('Rank')
plt.ylabel('Identification Accuracy')
plt.title('CMC Curve for subcategories')
plt.legend(leg)
plt.ylim(0,1.02)
plt.savefig('../Figures/cmc_subcategories.pdf')
plt.show()

plt.figure(figsize=(6,4))
plt.plot(np.tile(np.arange(1,K+1), (rss.shape[0],1)).T, rss.T)
plt.xticks(range(1,K+1))
plt.xlabel('Rank')
plt.ylabel('Recall')
plt.title('Recall at K for subcategories')
plt.legend(leg)
plt.ylim(0,1.02)
plt.savefig('../Figures/rekallatK_subcategories.pdf')
plt.show()
