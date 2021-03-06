import skimage.io
import skimage.transform
from PIL import ImageFile
import os
import ipdb

import numpy as np

def load_image( path, pre_height=146, pre_width=146, height=128, width=128 ):

    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    img /= 255.

    return (img * 2)-1

def crop_random(image_ori, width=176,height=176, x=None, y=None, overlap=7):
    if image_ori is None: return None
    
    image1 = image_ori.copy()
    crop = image_ori.copy()
    crop[40:40+height, 40:40+width] = 0
    image = np.zeros_like(image_ori)
    image[40-overlap:40+height+overlap,40-overlap:40+width+overlap] = image1[40-overlap:40+height+overlap,40-overlap:40+width+overlap]
    return image, crop, 0,0
