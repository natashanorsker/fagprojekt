#%%
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree
from sklearn import preprocessing
from torchvision import transforms
import itertools

from deepRanking.dataset import make_dataset
from deepRanking.nets import EmbeddingNet
from deepRanking.plots import extract_embeddings
import detectron2segment.inference
from utilities import dict_from_json

#%% Step 1 get embeddings for corpus
print('Getting embeddings')
model = EmbeddingNet()
model.load_state_dict(torch.load('deepRanking/models/online_model7-6_0.3979loss.pth', map_location=torch.device('cpu')))

catalog = dict_from_json('catalog.json')
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(catalog.keys()))

#make the 'normal' datasets:
# we limit the searchable database to X products to limit the time it takes to load it all in
train_dataset, test_dataset = make_dataset(label_encoder, n_test_products=100, NoDuplicates=True)

#make the dataloaders:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)


test_embeddings, test_labels = extract_embeddings(test_loader, model)
train_embeddings, train_labels = extract_embeddings(train_loader, model)
# concatenate to one 'database'
embeddings = np.concatenate((train_embeddings, test_embeddings),axis=0)
labels = np.concatenate((train_labels, test_labels))
dataset = np.concatenate((train_dataset, test_dataset),axis=0)

#%% Step 2 load an image and crop it out
print('Extracting jewellery')
impath = 'Figures/almas_ring.jpg'
im = cv2.imread(impath)

if im is None:
    # just makes it easier when you misstype a jpeg and jpg image
    if 'jpg' in impath:
        impath = impath[:-3]+'jpeg'
        im = cv2.imread(impath)
    elif 'jpeg' in impath:
        impath = impath[:-4]+'jpg'
        im = cv2.imread(impath)

crop_img = detectron2segment.inference.extractjewel(im)
# stupid cv2 nonsense they use BGR and not RGB like wtf? fixes image getting blue
crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) 
query_img = Image.fromarray(crop_img)
im = Image.open(impath)
#%% Step 2 testing. this is for proving that the same image is found
# testim = dataset[0][0].numpy()    
# testim = np.asarray(np.interp(testim, (0, 1), (0,255)),dtype=np.uint8)
# testim = np.swapaxes(testim,0,1)
# testim = np.swapaxes(testim,1,2)
# im = Image.fromarray(testim)
# query_img = Image.fromarray(testim)
#%% Step 3 get embeddings for new image
print('Searching for similar images')
model.eval()   # apparently does more than to print out the model. I think it freezes som weights or something

# currently images are squeezed quite a lot if they are not close to square this can be changed in the transform
# https://pytorch.org/vision/stable/transforms.html

transform = transforms.Compose([transforms.Resize((96,96)),
                                 transforms.ToTensor()])
img_t = transform(query_img)
batch_t = torch.unsqueeze(img_t, 0)

# Generate prediction
with torch.no_grad():
    embedding = model(batch_t)
    
#%% 4 find the nearest images
# this is a much more efficient datastructure than brute force search for nearest neighbours

tree = cKDTree(embeddings)
n_neighbor = 10
dists, idx = tree.query(embedding.numpy().ravel(), k=n_neighbor)

#%% 5 show  nearest images
fig, ax = plt.subplots(1, n_neighbor+2,figsize=((15, 4)))
fig.suptitle(f'{n_neighbor} Most similar images to query image',size='xx-large')

ax[0].imshow(im.resize((96*3,96*3)))
ax[0].axis('off')
ax[0].set_title('Query')

ax[1].imshow(query_img.resize((96*3,96*3)))
ax[1].axis('off')
ax[1].set_title('Extraction')
for i in range(n_neighbor):

    imgarr = dataset[idx.ravel()[i]][0].numpy() # ikke korrekt?
    imgarr = np.swapaxes(imgarr,0,1)
    imgarr = np.swapaxes(imgarr,1,2)
    rescaled_im = np.interp(imgarr, (imgarr.min(), imgarr.max()), (0,1))
    ax[i+2].imshow(rescaled_im)
    ax[i+2].set_title(f'dist: {dists.ravel()[i]:.3}')
    ax[i+2].axis('off')

plt.tight_layout()
plt.savefig('nearest.png',dpi=200)

print('Outputted to nearest.png')

# %%
