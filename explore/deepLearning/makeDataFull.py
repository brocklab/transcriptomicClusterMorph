# %% [markdown]
"""
# Making Training/Testing Data
Now that a suitable segmentation model has been developed, we should save individual images
of cells. 

In this notebook, I will load the model, apply it to every appropriate* image of monoculture
ESAM +/- cells, then save these images *with the background set to black*.\
\
*Appropriate meaning the cells is not significantly cut off by the edge, the cell is fluorescing
properly, and the date is appropriate.
"""

# %%
import sys, importlib
sys.path.append('../')
# importlib.reload(sys.modules['cellMorphHelper'])
import pickle
import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import rescale, resize

import matplotlib.pyplot as plt
import cellMorphHelper
import datetime

from skimage import data, measure
from skimage.segmentation import clear_border

import torch
import torch.nn.functional as F
# %%
def findFluorescenceColor(RGB, mask):
    """
    Finds the fluorescence of a cell
    Input: RGB image location
    Output: Color
    """
    # RGB = imread(RGBLocation)
    mask = mask.astype('bool')
    RGB[~np.dstack((mask,mask,mask))] = 0
    nGreen, BW = cellMorphHelper.segmentGreen(RGB)
    nRed, BW = cellMorphHelper.segmentRed(RGB)
    if nGreen>=(nRed+100):
        return "green"
    elif nRed>=(nGreen+100):
        return "red"
    else:
        return "NaN"
# %%
predictor = cellMorphHelper.getSegmentModel('../../output/TJ2201Split16')
# %% Filter out basics
experiment = 'TJ2201Split16'
finalDate = datetime.datetime(2022, 4, 8, 16, 0)
maxSize = 150
maxRows, maxCols = maxSize, maxSize
savePath = '../../data/esamMonoSegmented'

expPath = f'../data/{experiment}/'
pcPath = os.path.join(expPath, 'phaseContrast')
compositePath = os.path.join(expPath, 'composite')

pcIms = os.listdir(pcPath)
compositeIms = os.listdir(compositePath)
# Get rid of files not in appropriate well or after confluency date
imgBases = []
for pcFile in tqdm(pcIms):
    imgBase = cellMorphHelper.getImageBase(pcFile)
    well = imgBase.split('_')[0]
    date = cellMorphHelper.convertDate('_'.join(imgBase.split('_')[2:4]))
    if date < finalDate:
        imgBases.append(imgBase)
# %% Load and segment data
idx = 0
cellData = []
for imgBase in tqdm(imgBases):
    # Grab image
    well = imgBase.split('_')[0]
    pcFile = f'phaseContrast_{imgBase}.png'
    compositeFile = f'composite_{imgBase}.png'

    pcImg = imread(os.path.join(pcPath, pcFile))
    compositeImg = imread(os.path.join(compositePath, compositeFile))

    imSize = pcImg.shape
    outputs = predictor(pcImg)['instances'].to("cpu")
    nCells = len(outputs)

    # Go through each cell
    for cellNum in range(nCells):
        mask = outputs[cellNum].pred_masks.numpy()[0]

        # Crop to bounding box
        bb = list(outputs.pred_boxes[cellNum])[0].numpy()
        bb = [int(corner) for corner in bb]
        compositeCrop = compositeImg[bb[1]:bb[3], bb[0]:bb[2]].copy()
        pcCrop = pcImg[bb[1]:bb[3], bb[0]:bb[2]].copy()
        maskCrop = mask[bb[1]:bb[3], bb[0]:bb[2]].copy().astype('bool')
        color = findFluorescenceColor(compositeCrop, maskCrop)

        # pcCrop[~np.dstack((maskCrop,maskCrop,maskCrop))] = 0
        pcCrop = torch.tensor(pcCrop[:,:,0])
        # Keep aspect ratio and scale down data to be 150x150 (should be rare)
        if pcCrop.shape[0]>maxRows:
            pcCrop = rescale(pcCrop, maxRows/pcCrop.shape[0])
        if pcCrop.shape[1]>maxCols:
            pcCrop = rescale(pcCrop, maxRows/pcCrop.shape[1])

        # Now pad out the amount to make it 150x150
        diffRows = int((maxRows - pcCrop.shape[0])/2)+1
        diffCols = int((maxCols - pcCrop.shape[1])/2)
        pcCrop = F.pad(torch.tensor(pcCrop), pad=(diffCols, diffCols, diffRows, diffRows)).numpy()
        # Resize in case the difference was not actually an integer
        pcCrop = resize(pcCrop, (maxRows, maxCols))
        cellData.append([np.array(pcCrop), np.eye(2)[1]])
        # break
    break
#         # Save in appropriate folder
#         if well == 'E2' and color == 'red':
#             # saveFile = os.path.join(savePath, 'esamNegative', f'{imgBase}-{idx}.png')
#             # imsave(saveFile, pcCrop)
#             cellData.append([np.array(pcCrop), np.eye(2)[0]])
#             idx += 1
#         elif well == 'D2' and color == 'green':
#             # saveFile = os.path.join(savePath, 'esamPositive', f'{imgBase}-{idx}.png')
#             # imsave(saveFile, pcCrop)
#             cellData.append([np.array(pcCrop), np.eye(2)[1]])
#             idx += 1
# np.save('../data/esamMonoSegmented/cellUncrop.npy', cellData)
# %%
# bb = list(outputs.pred_boxes[4])[0].numpy()
# bb = [int(corner) for corner in bb]

# plt.figure(figsize=(10,30))
# plt.subplot(131)
# plt.imshow(compositeImg[bb[1]:bb[3], bb[0]:bb[2]])
# plt.title(bb)

# pixExpand = 20

# if bb[1] > pixExpand:
#     bb[1] = bb[1] - pixExpand
# if bb[3] < (imSize[0]-20):
#     bb[3] = bb[3] + pixExpand
# if bb[0] > pixExpand:
#     bb[0] = bb[0] - pixExpand
# if bb[2] < (imSize[1]-20):
#     bb[2] = bb[2] + pixExpand

# plt.subplot(132)
# plt.imshow(compositeImg[bb[1]:bb[3], bb[0]:bb[2]])
# plt.title(bb)

# plt.subplot(133)
# plt.imshow(compositeImg)

# # %% 
# import torch

# source = torch.tensor(pcCrop[:,:,0].copy())

# source_pad = F.pad(source, pad=(70,70, 70, 70))

# plt.imshow(source_pad)

# source_pad.shape

# # %%
# maxCols = 150
# pcCrop = torch.rand((160,900))
# pcCrop = torch.rand((160,170))