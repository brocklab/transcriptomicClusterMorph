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
from detectron2.data.datasets import convert_to_coco_json
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
predictor = cellMorphHelper.getSegmentModel('../models/TJ2201Split16', numClasses=1)
# %% Getting/reading *.npy
def getCells(experiment, predictor, stage=None):
    imDir = f'../data/{experiment}Split16/phaseContrast'
    allIms = os.listdir(imDir)
    wells = ['D2', 'E2', 'E7']
    # Filter out unwanted images for dates/time of confluency
    allIms = [im for im in allIms if im.split('_')[1] in wells]
    lastDate = datetime.datetime(2022, 4, 8, 16, 0)
    dates = [cellMorphHelper.convertDate('_'.join(im.split('_')[3:5])) for im in allIms]
    allIms = [im for im, date in zip(allIms, dates) if date<=lastDate]
    idx = 0

    # Generate training data/testing data
    if stage in ['train', 'test']:
        print(f'In {stage} stage')
        random.seed(1234)
        random.shuffle(allIms)
        trainPercent = 0.75
        trainNum = int(trainPercent*len(allIms))
        if stage == 'train':
            allIms = allIms[0:trainNum]
        elif stage == 'test':
            allIms = allIms[trainNum:]


    datasetDicts = []
    idx = 0
    for imFile in tqdm(allIms):
        imgPath = os.path.join(imDir, imFile)
        imgPathComposite = imgPath.replace('phaseContrast', 'composite')
        
        imgPc = imread(imgPath)
        imgComp = imread(imgPathComposite)

        record = {}
        record['file_name'] = imgPath
        record['image_id'] = idx
        record['height'] = imgPc.shape[0]
        record['width'] =  imgPc.shape[1]

        outputs = predictor(imgPc)
        outputs = outputs['instances']  
        masks = outputs.pred_masks.numpy()
        
        cells = []
        for mask in masks:
            actualColor = findFluorescenceColor(imgComp.copy(), mask.copy())
            if actualColor == 'red':
                classification = 0
            elif actualColor == 'green':
                classification = 1
            else:
                continue
            
            contours = measure.find_contours(mask.astype(np.uint8), .5)
            fullContour = np.vstack(contours)
            px = fullContour[:,1]
            py = fullContour[:,0]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]            

            cell = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classification,
            }
            cells.append(cell)
        record["annotations"] = cells
        datasetDicts.append(record)
        idx += 1          
    return datasetDicts

# %%
experiment = 'TJ2201'
stage = 'train'
if 'cellMorph_train' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph_train')
    MetadataCatalog.remove('cellMorph_train')

    print('Removing training')
if 'cellMorph_test' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph_test')
    MetadataCatalog.remove('cellMorph_test')

    print('Removing testing')
inputs = [experiment, predictor, 'train']

DatasetCatalog.register("cellMorph_train", lambda x=inputs: getCells(inputs[0], inputs[1], inputs[2]))
MetadataCatalog.get("cellMorph_" + "train").set(thing_classes=["esamNeg", "esamPos"])

DatasetCatalog.register("cellMorph_" + "test", lambda x=inputs: getCells(inputs[0], inputs[1], 'test'))
MetadataCatalog.get("cellMorph_" + "test").set(thing_classes=["esamNeg", "esamPos"])

# %% Write data
# convert_to_coco_json('cellMorph_train', '../data/cocoFiles/TJ2201Split16FullWellTrain.json', allow_cached=False)
convert_to_coco_json('cellMorph_test', '../data/cocoFiles/TJ2201Split16FullWellTest.json', allow_cached=False)
# %% Loading example
experiment = 'TJ2201'
stage = 'train'
if 'cellMorph_train' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph_train')
    MetadataCatalog.remove('cellMorph_train')

    print('Removing training')
if 'cellMorph_test' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph_test')
    MetadataCatalog.remove('cellMorph_test')

    print('Removing testing')

from detectron2.data.datasets import register_coco_instances
register_coco_instances("cellMorph_train", {}, '../data/cocoFiles/TJ2201Split16FullWellTrain.json', "")