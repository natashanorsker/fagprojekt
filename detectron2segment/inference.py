""" you cannot install detectron2 in windows, i recommend WSL on windows, should work on linux and macOS just fine though
the version used here only supports pytorch 1.7.x (i think, that what i used) you can check with. You don't need this unless you want to train it.
# print(torch.__version__, torch.cuda.is_available())

built using detectron2

I used detectron v0.3. just note that the version of pytorch and detectron need to be compatible see more here https://detectron2.readthedocs.io/en/latest/tutorials/install.html. Can be installed like so
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html

if you want to train a new version of the classifier or improve, use the colab notebook listed here
https://colab.research.google.com/drive/1JVKUjI4ksjJCX9K0oAqIhx22Hr1Xbf5d?usp=sharing
it takes ~5 min to train on the gpu in colab. That is why I trained it there

Original code
https://github.com/facebookresearch/detectron2

you need to have the model_final.pth downloaded and placed in the detectron2segment folder. it is too large for github. sorry. It is on the drive
"""
#import torch, torchvision

# import some common libraries
import numpy as np
import os, cv2
import warnings
# i know this is a bad bad idea but hey ＼（〇_ｏ）／
warnings.filterwarnings("ignore")


# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger
setup_logger()



def extract(im, threshold=0.82):
    """
    
    extracts All jewels from an image, crops the image to fit the jewel and makes all non-jewel elements in the image white.

    >>> input a cv2 image (PIL does not seems to work)
    >>> returns the cropped image
    
    example use if called from root directory
    from detectron2segment.inference import extract
    import cv2

    crop_img = extract('detectron2segment/test.jpeg')
    cv2.imwrite('extracted3.png',crop_img)
    """
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join('detectron2segment','config.yml'))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = os.path.join('detectron2segment', 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set a custom testing threshold

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im) 
    v = Visualizer(im[:, :, ::-1], 
                    scale=1,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    numjewel = len(outputs["instances"].pred_masks.cpu())*1
    mask2 = np.zeros(np.shape(im)[:2])
    box = np.zeros((numjewel,4))
    for i in range(numjewel):
        mask2 = mask2 + np.asarray(outputs["instances"].pred_masks[i].cpu())*1
        box[i,:] = np.asarray(outputs["instances"].pred_boxes[i].to('cpu').tensor[0],dtype=int)
    mask3d = np.dstack((np.dstack((mask2, mask2)),mask2))*1
    # mask by multiplication, clip to range 0 to 255 and make integer
    result2 = (im * mask3d).clip(0, 255).astype(np.uint8)
    result2[mask3d==0] = 255
    low1 = int(np.min(box[:,1]))
    low2 = int(np.min(box[:,0]))
    high1 = int(np.max(box[:,3]))
    high2 = int(np.max(box[:,2]))
    crop_img = result2[low1:high1, low2:high2]
    return crop_img


def extract1jewel(im, threshold=0.82):
    """
    extracts 1 jewel from an image this can be a bit random if it by chance misclassifies something, crops the image to fit the jewel and makes all non-jewel elements in the image white.

    
    example use if called from root directory
    from detectron2segment.inference import extract
    import cv2

    crop_img = extract('detectron2segment/test.jpeg')
    cv2.imwrite('extracted3.png',crop_img)
    """
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join('detectron2segment','config.yml'))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = os.path.join('detectron2segment', 'model_final.pth')
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im) 

    v = Visualizer(im[:, :, ::-1], 
                    scale=1,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    mask2 = np.asarray(outputs["instances"].pred_masks[0].cpu())*1
    mask3d = np.dstack((np.dstack((mask2, mask2)),mask2))*1
    # mask by multiplication, clip to range 0 to 255 and make integer
    result2 = (im * mask3d).clip(0, 255).astype(np.uint8)
    result2[mask3d==0] = 255
    box = np.asarray(outputs["instances"].pred_boxes[0].to('cpu').tensor[0],dtype=int)
    crop_img = result2[box[1]:box[3], box[0]:box[2]]
    return crop_img


if __name__ == "__main__":
    im = cv2.imread(os.path.join('detectron2segment','test.jpeg'))
    crop_img = extract(im, threshold=0.82)
    cv2.imwrite('extracted.png',crop_img)
