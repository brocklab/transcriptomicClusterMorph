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
predictorClassify = cellMorphHelper.getSegmentModel('../models/TJ2201Split16Classify', numClasses=2)
# %%
makeData = 0
if makeData:
    # Grab all images associated with relevant monoculture wells
    wells = ['E2', 'D2']
    esamPosWells = ['E2']
    esamNegWells = ['D2']
    wellClass = {'E2': 0, 'D2': 1}
    allIms = os.listdir('../data/TJ2201Split16/phaseContrast')
    allIms = [im for im in allIms if im.split('_')[1] in wells]
    dateAccuracy = {well: {} for well in wells}
    # For each image calculate accuracy
    for i, im in enumerate(tqdm(allIms)):
        date = '_'.join(im.split('_')[3:5])
        well = im.split('_')[1]
        if date not in dateAccuracy[well].keys():
            dateAccuracy[well][date] = [0, 0]
        imPath = os.path.join('../data/TJ2201Split16/phaseContrast', im)
        outputs = predictorClassify(imread(imPath))
        outputs = outputs['instances']

        desiredClass = wellClass[well]

        predictions = outputs.pred_classes.numpy()
        dateAccuracy[well][date][0] += sum(predictions == desiredClass)
        dateAccuracy[well][date][1] += len(predictions)   

        if i%10 == 0:
            pickle.dump(dateAccuracy, open('../data/longitudinalAccuracy.pickle', "wb"))

    pickle.dump(dateAccuracy, open('../data/longitudinalAccuracy.pickle', "wb"))
# %%
longitudinalAccuracy=pickle.load(open('../data/longitudinalAccuracy.pickle',"rb"))

# First, convert all items to accuracies
E2 = longitudinalAccuracy['E2']
E2Dates, E2Acc = [], []
for date in E2.keys():
    E2Acc.append(E2[date][0]/E2[date][1])
    E2Dates.append(cellMorphHelper.convertDate(date))
D2 = longitudinalAccuracy['D2']
D2Dates, D2Acc = [], []
for date in D2.keys():
    D2Acc.append(D2[date][0]/D2[date][1])
    D2Dates.append(cellMorphHelper.convertDate(date))

plt.figure(figsize=(10,5))
plt.scatter(D2Dates, D2Acc, c='green', label = 'D2: ESAM (+)')
plt.scatter(E2Dates, E2Acc, c='red', label = 'E2: ESAM (-)')
plt.xlabel('Date of Image Capture')
plt.ylabel('Accuracy of Identified Cells')
plt.legend()
# %%
desiredDate = ['2022y04m07d_12h00m']

# Grab all images associated with relevant monoculture wells
wells = ['D2']
esamPosWells = ['E2']
esamNegWells = ['D2']
wellClass = {'E2': 0, 'D2': 1}
allIms = os.listdir('../data/TJ2201Split16/phaseContrast')
allIms = [im for im in allIms if im.split('_')[1] in wells and '_'.join(im.split('_')[3:5]) in desiredDate]
dateAccuracy = {well: {} for well in wells}
# %%
well = 'D2'
# For each image calculate accuracy
for i, im in enumerate(tqdm(allIms)):
    date = '_'.join(im.split('_')[3:5])
    imNum = im.split('_')[2]
    if imNum not in dateAccuracy[well].keys():
        dateAccuracy[well][imNum] = [0, 0]
    imPath = os.path.join('../data/TJ2201Split16/phaseContrast', im)
    outputs = predictorClassify(imread(imPath))
    outputs = outputs['instances']

    desiredClass = wellClass[well]

    predictions = outputs.pred_classes.numpy()
    dateAccuracy[well][imNum][0] += sum(predictions == desiredClass)
    dateAccuracy[well][imNum][1] += len(predictions)   