# %% [markdown]
"""
# Making Training/Testing Data
Now that a suitable segmentation model has been developed, we should save individual images
of cells. 

In this notebook, I will load the model, apply it to every split image of monoculture
and save the segmentation. 
# TODO: Generalize this?
# TODO: Fix 
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

from detectron2.structures import BoxMode
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
predictor = cellMorphHelper.getSegmentModel('../../models/TJ2201Split16')
# %% Filter out basics
experiment = 'TJ2201'
finalDate = datetime.datetime(2022, 4, 8, 16, 0)
maxSize = 150
maxRows, maxCols = maxSize, maxSize

expPath = f'../../data/{experiment}Split16/'
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
phenoCounts = {'monoPos': 0, 'monoNeg': 0, 'coNeg': 0, 'coPos': 0}
# %%
n = 200
imgBase = imgBases[n]

pcFile = f'phaseContrast_{imgBase}.png'

pcImg = imread(os.path.join(pcPath, pcFile))

pcImg.shape
# %% Load and segment data
maxCount = 5000

if os.path.isfile(f'./{experiment}DatasetDict.npy'):
    datasetDicts = list(np.load(f'./{experiment}DatasetDict.npy', allow_pickle=True))
    processedFiles = [img['file_name'] for img in datasetDicts]
    idx = max([img['image_id'] for img in datasetDicts])+1
else:
    datasetDicts = []
    idx = 0

categoryDict = {'green': 0, 'red': 1}
for imgBase in tqdm(imgBases, leave=True):
    # Grab image
    well = imgBase.split('_')[0]
    # Early stopping
    # if well in monoPos or well in monoNeg:
    #     if phenoCounts[wellTypes[well]] > maxCount:
    #         continue
    pcFile = f'phaseContrast_{imgBase}.png'
    compositeFile = f'composite_{imgBase}.png'
    pcFileFull = os.path.join(pcPath, pcFile)
    compositeFileFull = os.path.join(compositePath, compositeFile)

    if pcFileFull in processedFiles:
        continue
    pcImg = imread(pcFileFull)
    compositeImg = imread(compositeFileFull)

    # Make sure it's a grayscale image
    # if len(pcImg.shape)>2:
    #     pcImg = rgb2gray(pcImg)




    outputs = predictor(pcImg)['instances'].to("cpu")
    nCells = len(outputs)
    # if nCells == 0:
    #     continue
    # Go through each cell in each cropped image
    record = {}
    record['file_name'] = pcFileFull
    record['image_id'] = idx
    record['height'] = pcImg.shape[0]
    record['width'] =  pcImg.shape[1]

    cells = []
    for cellNum in range(nCells):
        mask = outputs[cellNum].pred_masks.numpy()[0]
        color = findFluorescenceColor(compositeImg, mask)
        if color not in ['red', 'green']:
            continue
        contours = measure.find_contours(mask, .5)
        fullContour = np.vstack(contours)

        px = fullContour[:,1]
        py = fullContour[:,0]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        
        bbox = list(outputs[cellNum].pred_boxes.tensor.numpy()[0])


        cell = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": categoryDict[color],
        }

        cells.append(cell)
    record["annotations"] = cells
    datasetDicts.append(record)
    idx += 1

    if idx % 100 == 0:
        np.save(f'./{experiment}DatasetDict.npy', datasetDicts)
        
# %% Test datasetDicts
# d = datasetDicts[4]
# img = imread(d['file_name'])
# p = d['annotations'][0]['segmentation']
# plt.imshow(img)

# p1 = [pt for i, pt in enumerate(p[0]) if i%2 == 0]
# p2 = [pt for i, pt in enumerate(p[0]) if i%2 != 0]

# plt.plot(p1, p2)
# %% Old code for saving images
# # Crop to bounding box
# bb = list(outputs.pred_boxes[cellNum])[0].numpy()
# bb = [int(corner) for corner in bb]
# compositeCrop = compositeImg[bb[1]:bb[3], bb[0]:bb[2]].copy()
# pcCrop = pcImg[bb[1]:bb[3], bb[0]:bb[2]].copy()
# maskCrop = mask[bb[1]:bb[3], bb[0]:bb[2]].copy().astype('bool')
# color = findFluorescenceColor(compositeCrop, maskCrop)

# pcCrop[~np.dstack((maskCrop,maskCrop,maskCrop))] = 0
# pcCrop = torch.tensor(pcCrop[:,:,0])
# # Keep aspect ratio and scale down data to be 150x150 (should be rare)
# if pcCrop.shape[0]>maxRows:
#     pcCrop = rescale(pcCrop, maxRows/pcCrop.shape[0])
# if pcCrop.shape[1]>maxCols:
#     pcCrop = rescale(pcCrop, maxRows/pcCrop.shape[1])

# # Now pad out the amount to make it 150x150
# diffRows = int((maxRows - pcCrop.shape[0])/2)+1
# diffCols = int((maxCols - pcCrop.shape[1])/2)
# pcCrop = F.pad(torch.tensor(pcCrop), pad=(diffCols, diffCols, diffRows, diffRows)).numpy()
# # Resize in case the difference was not actually an integer
# pcCrop = resize(pcCrop, (maxRows, maxCols))

# # Determine save folder
# saveFlag = 0
# if wellTypes[well] == 'monoPos' and color == 'green':
#     phenoFolder = wellTypes[well]
#     saveFlag = 1
#     phenoCounts[phenoFolder] += 1
# elif wellTypes[well] == 'monoNeg' and color == 'red':
#     phenoFolder = wellTypes[well]
#     saveFlag = 1
#     phenoCounts[phenoFolder] += 1
# elif wellTypes[well] == 'co' and color == 'green':
#     phenoFolder = wellTypes[well]+'Pos'
#     saveFlag = 1
#     phenoCounts[phenoFolder] += 1
# elif wellTypes[well] == 'co' and color == 'red':
#     phenoFolder = wellTypes[well]+'Neg'
#     saveFlag = 1
#     phenoCounts[phenoFolder] += 1


# saveFile = os.path.join(savePath, phenoFolder, f'{imgBase}-{idx}.png')
# if saveFlag and phenoCounts[phenoFolder] <= maxCount:
#     break
#     imsave(saveFile, pcCrop)
