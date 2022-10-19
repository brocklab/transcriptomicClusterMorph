# %%
import sys
sys.path.append('../scripts')
import cellMorphHelper

import os
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
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
predictor = cellMorphHelper.getSegmentModel('../output/TJ2201Split16ClassifyFull', numClasses=2)
# %%
if 'cellMorph_test' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph_test')
    MetadataCatalog.remove('cellMorph_test')

    print('Removing testing')
register_coco_instances("cellMorph_test", {}, '../data/cocoFiles/TJ2201Split16FullWellTest.json', "")
# %%
images = DatasetCatalog['cellMorph_test']()
fileNames = [image['file_name'] for image in images]
wells = [fileName.split('_')[1] for fileName in fileNames]

imgLog = {'D2': {}, 'E2': {}, 'E7': {}}
for image in tqdm(images):
    fileName = image['file_name']
    well = fileName.split('_')[1]
    # For each well/file log the number correct and the number of cells total
    imgLog[well][fileName] = [0, 0]
    annotations = image['annotations']
    if len(annotations) == 0:
        continue
    # Load images and find results from predictor
    imgPc = imread(fileName)
    imgComp = imread(fileName.replace('phaseContrast', 'composite'))
    outputs = predictor(imgPc)
    outputs = outputs['instances']  
    masks = outputs.pred_masks.numpy()
    pred_classes = outputs.pred_classes.numpy()
    # For each mask, compare its identity to the predicted class
    for mask, pred_class in zip(masks, pred_class):
        actualColor = findFluorescenceColor(imgComp.copy(), mask.copy())
        if actualColor == 'red':
            classification = 0
        elif actualColor == 'green':
            classification = 1
        else:
            continue
    
        if classification == pred_class:
            imgLog[well][fileName][0] += 1
        imgLog[well][fileName][1] +=1
