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

# TODO: Generalize this?
"""

# %%
import sys, importlib
sys.path.append('../../scripts')
# importlib.reload(sys.modules['cellMorphHelper'])
import pickle
import os
import cv2
import numpy as np
import random
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

expPath = f'../../data/{experiment}/'
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
        
random.seed(1234)
random.shuffle(imgBases)
# %% Define structure and phenotype to well

# Make save structure
savePath = f'../../data/{experiment}SingleCell'
os.makedirs(savePath, exist_ok=True)
os.makedirs(f'{savePath}/monoPos', exist_ok=True)
os.makedirs(f'{savePath}/monoNeg', exist_ok=True)
os.makedirs(f'{savePath}/coPos', exist_ok=True)
os.makedirs(f'{savePath}/coNeg', exist_ok=True)

# 
monoPos = ['B2','B3','B4','B5','B6','C2','C3','C4','C5','C6','D2','D3','D4','D5','D6']
monoNeg = ['E2','E3','E4','E5','E6','F2','F3','F4','F5','F6','G2','G3','G4','G5','G6']
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
wellTypes = {}
for well in monoPos:
    wellTypes[well] = 'monoPos'
for well in monoNeg:
    wellTypes[well] = 'monoNeg'
for well in co:
    wellTypes[well] = 'co'
phenoCounts = {'monoPos': 0, 'monoNeg': 0, 'co': 0}
# %% Load and segment data
maxCount = 100000
idx = 0
usedBases = []
for imgBase in tqdm(imgBases):
    # Grab image
    well = imgBase.split('_')[0]
    print(well)
    # Early stopping
    if phenoCounts[wellTypes[well]] > maxCount:
        continue
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

        pcCrop[~np.dstack((maskCrop,maskCrop,maskCrop))] = 0
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
        
        # Determine save folder
        saveFlag = 0
        if wellTypes[well] == 'monoPos' and color == 'green':
            phenoFolder = wellTypes[well]
            saveFlag = 1
        elif wellTypes[well] == 'monoNeg' and color == 'red':
            phenoFolder = wellTypes[well]
            saveFlag = 1
        elif wellTypes[well] == 'co' and color == 'green':
            phenoFolder = wellTypes[well]+'Pos'
            saveFlag = 1
        elif wellTypes[well] == 'co' and color == 'red':
            phenoFolder = wellTypes[well]+'Neg'
            saveFlag = 1
        

        saveFile = os.path.join(savePath, phenoFolder, f'{imgBase}-{idx}.png')
        print(saveFile)
        break
        if saveFlag:
            imsave(saveFile, pcCrop)
        

        usedBases.append(imgBase)