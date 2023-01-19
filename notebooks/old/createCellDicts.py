# %%
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
import os, json, cv2, random
import matplotlib.pyplot as plt
import datetime

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.io import imread
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
from scipy.spatial import ConvexHull

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
# %% Functions
def clearEdgeCells(mask):
    """
    Checks if cells are on border by dilating them and then clearing the border. 
    NOTE: This could be problematic since some cells are just close enough, but could be solved by stitching each image together, then checking the border.
    
    Returns 0 if mask is on edge, else returns 1
    """
    maskDilate = binary_dilation(mask)
    maskFinal = clear_border(maskDilate)
    if np.sum(maskFinal)==0:
        return 0
    else:
        return 1

def findFluorescenceColor(RGB, mask):
    """
    Finds the fluorescence of a cell
    Input: RGB image location
    Output: Color
    """
    RGB[~np.dstack((mask,mask,mask))] = 0
    nGreen, BW = cellMorphHelper.segmentGreen(RGB)
    nRed, BW = cellMorphHelper.segmentRed(RGB)
    if nGreen>=(nRed+100):
        return "green"
    elif nRed>=(nGreen+100):
        return "red"
    else:
        return "NaN"

# %% Data loader
expDir = 'TJ2201Split16'
# def getCellDicts(expDir, stage):
imgDir = os.path.join('../data/',expDir,'phaseContrast')
imgs = os.listdir(imgDir)
# %%
datasetDicts = []
idx = 0

for img in imgs:
    imgBase = cellMorphHelper.getImageBase(img)
    date = '_'.join(imgBase.split('_')[2:-1])
    if date<datetime.datetime(2022, 4, 8, 16, 0):
        continue

    maskName = 'mask_'+imgBase
    compositeName = 'composite_'+imgBase
    phaseContrastPath =  os.path.join(imgDir, img)
    maskPath = os.path.join('../data', expDir, 'mask', maskName+'.tif')
    imgMask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
    compositePath = os.path.join('../data/',expDir,'composite', compositeName+'.png')
    compositeIm = imread(compositePath)
    height, width = imgMask.shape[:2]
    cellMorphHelper.printProgressBar(idx, len(imgs), prefix = 'Loading information')
    record = {}

    record['file_name'] = phaseContrastPath
    record['image_id'] = idx
    record['height'] = height
    record['width'] = width   

    maskLabels = np.unique(imgMask)
    maskLabels = maskLabels[maskLabels!=0]
    for maskLabel in maskLabels:
        cellMask = np.array(imgMask==maskLabel).astype('uint8')

        fluorescenceColor = findFluorescenceColor(compositeIm, cellMask)
        if fluorescenceColor == 'NaN':
            continue

        contours = measure.find_contours(cellMask, .5)
        contours = np.vstack(contours)
        hull = contours

        # if clearEdgeCells(cellMask)==0:
        #     plt.figure()
        #     plt.imshow(imgMask)
        #     hull = np.array(hull)
        #     px = hull[:,1]
        #     py = hull[:,0]
        #     plt.scatter(px, py, c='red',s=0.5)
        #     continue
    

        px = hull[:,1]
        py = hull[:,0]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]

        obj = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": 0,
        }
        objs.append(obj)
    record["annotations"] = objs
    datasetDicts.append(record)
    if idx % 100 == 0:
        pickle.dump(datasetDicts, open(f'{expDir}_datasetDict.pickle', "wb"))
    objs = []
    idx+=1

pickle.dump(datasetDicts, open(f'{expDir}_datasetDict.pickle', "wb"))
