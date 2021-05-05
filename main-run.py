import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree
from sklearn import preprocessing
from torchvision import transforms

from deepRanking.dataset import make_dataset
from deepRanking.nets import EmbeddingNet
from deepRanking.plots import extract_embeddings, plot_embeddings
from detectron2segment.inference import extract, extract1jewel
from utilities import dict_from_json

#%% Step 1 load an image and crop it out
print('Extracting jewellery')
impath = 'Figures/test3.jpg'
im = cv2.imread(impath)
crop_img = extract1jewel(im)
# stupid cv2 nonsense they use BGR and not RGB like wtf? fixes image getting blue
crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) 
query_img = Image.fromarray(crop_img)
im = Image.open(impath)
#%% Step 2 get embeddings for corpus
print('Getting embeddings')
model = EmbeddingNet()
model.load_state_dict(torch.load('deepRanking/online_model.pth'))

label_encoder = preprocessing.LabelEncoder()
catalog = dict_from_json('catalog.json')
label_encoder.fit(list(catalog.keys()))

#make the 'normal' datasets:
train_dataset, test_dataset = make_dataset(label_encoder, n_test_products=100)

#make the dataloaders:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False)

# extract embeddings and plot:
val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_tl, val_labels_tl, encoder=label_encoder)

#%% Step 3 get embeddings for new image
print('Searching for similar images')
model.eval()   # apparently does more than to print out the model. I think it freezes som weights or something

# currently images are squeezed quite a lot if they are not close to square this can be changed in the transform
# https://pytorch.org/vision/stable/transforms.html
transform = transforms.Compose([
                transforms.Resize((96,96)),           
                transforms.ToTensor(),                     
                transforms.Normalize(                      
                mean=[0.485, 0.456, 0.406],                
                std=[0.229, 0.224, 0.225]                  
 )])

img_t = transform(query_img)
batch_t = torch.unsqueeze(img_t, 0)

# Generate prediction
with torch.no_grad():
    embedding = model(batch_t)
    
#%% 4 find the nearest images
# this is a much more efficient datastructure than brute force search for nearest neighbours

tree = cKDTree(val_embeddings_tl)
n_neighbor = 5
dists, idx = tree.query(embedding.numpy().ravel(), k=n_neighbor)

#%% 5 show  nearest images
fig, ax = plt.subplots(1, n_neighbor+2,figsize=((15, 4)))
fig.suptitle('5 Most similar images to query image',size='xx-large')

ax[0].imshow(im.resize((96*3,96*3)))
ax[0].axis('off')
ax[0].set_title('Query')

ax[1].imshow(query_img.resize((96*3,96*3)))
ax[1].axis('off')
ax[1].set_title('Extraction')
for i in range(n_neighbor):
    
    imgarr = test_dataset[idx.ravel()[i]][0].numpy()
    imgarr=np.swapaxes(imgarr,0,1)
    imgarr=np.swapaxes(imgarr,1,2)
    ax[i+2].imshow(np.interp(imgarr, (imgarr.min(), imgarr.max()), (0,1)))
    ax[i+2].set_title(f'dist: {dists.ravel()[i]:.3}')
    ax[i+2].axis('off')

plt.tight_layout()
plt.savefig('nearest.png',dpi=200)

print('Outputted to nearest.png')

# Show result
# plt.imshow(image, cmap='gray')
# plt.savefig('image')
# plt.show()


# %%
