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
predictorClassify = cellMorphHelper.getSegmentModel('../output/TJ2201Split16Classify', numClasses=2)
# %%
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
        pickle.dump(dateAccuracy, open('../data/longitudinalAccuracy', "wb"))

pickle.dump(dateAccuracy, open('../data/longitudinalAccuracy', "wb"))
# %%