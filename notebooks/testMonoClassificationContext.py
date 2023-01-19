# %% [markdown]
"""
This is a notebook meant to test the network's ability to classify cells. 
This is done by:
1. Demonstrating prediction capability in mono and coculture
2. Demonstrating prediction capability in pseudo-culculture, ie manually grabbing images from 
"""
# %%
import sys
sys.path.append('../scripts')
import cellMorphHelper

import torch
import detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random, pickle
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.io import imread
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
from scipy.spatial import ConvexHull

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
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
# %% Get Predictor
predictorSegment = cellMorphHelper.getSegmentModel('../models/TJ2201Split16', numClasses=1)
predictorClassify = cellMorphHelper.getSegmentModel('../models/TJ2201Split16ClassifyFull', numClasses=2)
# %% Get images/known segmentations

# Set information for desired experiment/well
experiment = 'TJ2201'
stage = 'test'
# wells = ['D2', 'E2']

# Grab segmentations
segDir = os.path.join('../data',experiment,'segmentedIms')
segFiles = os.listdir(segDir)
segFiles = [segFile for segFile in segFiles if segFile.endswith('.npy')]
# Get training data
random.seed(1234)
random.shuffle(segFiles)
trainPercent = 0.75
trainNum = int(trainPercent*len(segFiles))
if stage == 'train':
    segFiles = segFiles[0:trainNum]
elif stage == 'test':
    segFiles = segFiles[trainNum:]

# Get image files
testIms = {}
pcDir = f'../data/{experiment}Split16/phaseContrast'
for segFile in segFiles:
    seg = np.load(os.path.join(segDir, segFile), allow_pickle=True)
    seg = seg.item()
    imgBase = cellMorphHelper.getImageBase(seg['filename'].split('/')[-1])
    well = imgBase.split('_')[0]
    if well not in testIms.keys():
        testIms[well] = []
    else:
        for splitNum in range(1, 17):
            testIms[well].append(f'{pcDir}/phaseContrast_{imgBase}_{splitNum}.png')
# %% Show monoculture ability - ESAM (-)
cellMorphHelper.viewPredictorResult(predictorClassify, testIms['E2'][0])
# %% Show monoculture ability - ESAM (+)
cellMorphHelper.viewPredictorResult(predictorClassify, testIms['D2'][12])
# %% Show coculture  ability
cellMorphHelper.viewPredictorResult(predictorClassify, testIms['E7'][1])
plt.imshow(imread(testIms['E7'][1].replace('phaseContrast', 'composite')))
# %% Calculate accuracy in monoculture
nCorrect, nTotal = 0, 0
for img in tqdm(testIms['E2']):
    outputs = predictorClassify(imread(img))
    predictions = outputs['instances'].pred_classes.numpy()
    nCorrect += sum(predictions == 0)
    nTotal += len(predictions)
print(f'Well E2: {nCorrect/nTotal:0.2f}')

nCorrect, nTotal = 0, 0
for img in tqdm(testIms['D2']):
    outputs = predictorClassify(imread(img))
    predictions = outputs['instances'].pred_classes.numpy()
    nCorrect += sum(predictions == 1)
    nTotal += len(predictions)
print(f'Well D2: {nCorrect/nTotal:0.2f}')

# %% Calculate accuracy in coculture
predicted, actual = [], []
nCorrect, nTotal = 0, 0
for img in tqdm(testIms['E7']):
    pcImg = imread(img)
    compositeImg = imread(img.replace('phaseContrast', 'composite'))
    outputs = predictorClassify(pcImg)
    outputs = outputs['instances']
    masks = outputs.pred_masks.numpy()
    pred_classes = outputs.pred_classes.numpy()

    for mask, pred_class in zip(masks, pred_classes):
        actualColor = findFluorescenceColor(compositeImg.copy(), mask.copy())
        if actualColor == 'red':
            classification = 0
        elif actualColor == 'green':
            classification = 1
        else:
            continue
        
        predicted.append(pred_class)
        actual.append(classification)
        if classification == pred_class:
            nCorrect+=1
        nTotal += 1
# %%
