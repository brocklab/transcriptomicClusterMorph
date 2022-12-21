# %%
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from skimage.io import imread
from skimage.transform import resize

# from singleCellLoader import singleCellCrop
import torch
import torch.nn.functional as F

# %%
experiment = 'TJ2201'
dataPath = f'../../data/{experiment}/{experiment}Split16/phaseContrast'
datasetDicts = np.load(f'./{experiment}DatasetDict.npy', allow_pickle=True)
# %%
segmentations, phenotypes, imgPaths, bbs = [], [], [], []
# Note there is a lot of repeats for images but this is much cleaner
for img in datasetDicts:
    path = img['file_name'].split('/')[-1]
    for annotation in img['annotations']:
        segmentations.append(np.array(annotation['segmentation'][0]))
        phenotypes.append(annotation['category_id'])
        imgPaths.append(os.path.join(dataPath, path))
        bbs.append([int(corner) for corner in annotation['bbox']])
# %% Only crop to bounding box
idx = 1
# idx = 1
# '../../data/TJ2201/TJ2201Raw/phaseContrast/phaseContrast_F4_9_2022y04m05d_20h00m.png'
nIncrease = 0
maxImgSize = 150


imgName = imgPaths[idx]
label = phenotypes[idx]
maxRows, maxCols = maxImgSize, maxImgSize
img = imread(imgName)

bb = bbs[idx]

nIncrease = nIncrease
colMin, rowMin, colMax, rowMax = bb
rowMin -= nIncrease
rowMax += nIncrease
colMin -= nIncrease
colMax += nIncrease

# Indexing checks
if rowMin <= 0:
    rowMin = 0
if rowMax > img.shape[0]:
    rowMax = img.shape[0]
if colMin <= 0:
    colMin = 0
if colMax >= img.shape[1]:
    colMax = img.shape[1]

# Increase the size of the bounding box and crop
bbIncrease = [colMin, rowMin, colMax, rowMax]
imgCrop = img[bbIncrease[1]:bbIncrease[3], bbIncrease[0]:bbIncrease[2]]

# Pad image
diffRows = int((maxRows - imgCrop.shape[0])/2)
diffCols = int((maxCols - imgCrop.shape[1])/2)
pcCrop = F.pad(torch.tensor(imgCrop[:,:,0]), pad=(diffCols, diffCols, diffRows, diffRows)).numpy()
pcCrop = resize(pcCrop, (maxRows, maxCols))

plt.imshow(pcCrop, cmap='gray')