# %%
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics
import random

from skimage.measure import label
from skimage.color import label2rgb
from skimage.io import imread
# %%
model_path = '../models/cellPoseModels/CP_20220907_103108'
model = models.CellposeModel(gpu=0, pretrained_model=model_path)
# %%
imPath = '../data/TJ2201/composite'
files = [os.path.join(imPath, file) for file in os.listdir(imPath) if file.endswith('.png')]
nIm = 1
nIms = len(files)
for file in files:
    print(f'{nIm}/{nIms}')
    image = imread(file)
    mask, flow, style = model.eval(image, channels = [0, 2])
    io.masks_flows_to_seg(image, mask, flow, model.diam_labels, file_names = file)
    nIm+=1