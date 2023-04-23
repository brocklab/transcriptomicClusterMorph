# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.data.fileManagement import convertDate, splitName2Whole
from src.data.imageProcessing import split2WholeCoords, expandImageSegmentation
from src.models import modelTools
from pathlib import Path
import numpy as np
import time
import sys
import datetime
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.draw import polygon
import cv2
from matplotlib.path import Path as matPath

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim
print('Imported everything')


# %%
experiment = 'TJ2201'
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
# %%
# Get index of dataset dict with cell
idx = np.random.randint(len(datasetDicts))
while True:
    if len(datasetDicts[idx]['annotations']) != 0:
        break
    else:
        idx += 1
print(idx)

idx = 11451
poly = datasetDicts[idx]['annotations'][0]['segmentation'][0]
imgName = datasetDicts[idx]['file_name']
bb = datasetDicts[idx]['annotations'][0]['bbox']
nIms = 16
imgNameWhole = splitName2Whole(Path(imgName).name)
imgWhole = imread(Path('../data/TJ2201/raw/phaseContrast', imgNameWhole))
nIncrease = 50
padNum = 200

# bbIncrease(poly, bb, imgName, imgWhole, nIms, nIncrease=50, padNum=200)
from skimage.draw import polygon2mask

"""
Takes in a segmentation from a split image and outputs the segmentation from the whole image. 
Inputs: 
- poly: Polygon in datasetDict format
- bb: Bounding box in datasetDict format
- imageName: Name of the image where the segmentation was found
- imgWhole: The whole image from which the final crop will come from
- nIncrease: The amount to increase the bounding box
- padNum: The padding on the whole image, necessary to segment properly

Outputs:
- imgBBWholeExpand: The image cropped from the whole image increased by nIncrease
"""
splitNum = int(imgName.split('_')[-1].split('.')[0])
coords = split2WholeCoords(nIms, wholeImgSize = imgWhole.shape)
imgWhole = np.pad(imgWhole, (padNum,padNum))
polyxWhole, polyyWhole, bbWhole = expandImageSegmentation(poly, bb, splitNum, coords, padNum)
bbWhole = [int(corner) for corner in bbWhole]
colMin, rowMin, colMax, rowMax = bbWhole
rowMin -= nIncrease
rowMax += nIncrease
colMin -= nIncrease
colMax += nIncrease

maskBlackout  = polygon2mask(imgWhole.shape, np.array([polyyWhole, polyxWhole], dtype="object").T)

imgWhole[maskBlackout] = 0

bbIncrease = [colMin, rowMin, colMax, rowMax]
imgBBWholeExpand = imgWhole[bbIncrease[1]:bbIncrease[3], bbIncrease[0]:bbIncrease[2]]

plt.imshow(imgBBWholeExpand)
# %%



plt.imshow(imgWholeBlackout)
# %%
plt.figure(figsize=(50,50))
plt.imshow(imgWhole, cmap='gray')
plt.plot(polyxWhole, polyyWhole)

# %%
width, height=2000, 2000

polygon=[(0.1*width, 0.1*height), (0.15*width, 0.7*height), (0.8*width, 0.75*height), (0.72*width, 0.15*height)]
poly_path=matPath(polygon)

x, y = np.mgrid[:height, :width]
coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)

mask = poly_path.contains_points(coors)
plt.imshow(mask.reshape(height, width))
plt.show()
# %%

bbIncrease = [colMin, rowMin, colMax, rowMax]
imgBBWholeExpand = imgWhole[bbIncrease[1]:bbIncrease[3], bbIncrease[0]:bbIncrease[2]]
plt.imshow(imgBBWholeExpand)

