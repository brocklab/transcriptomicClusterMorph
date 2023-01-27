# %% [markdown]
"""
This file stores segmentation tools
"""

# %%
from src.data.imageProcessing import imSplit
from src.data.fileManagement import getImageBase

import numpy as np
import os
import random
from tqdm import tqdm

from skimage import measure
from skimage import img_as_float

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
# %%
def cellpose2Detectron(experiment, imgType='phaseContrast', stage=None):
    """
    Takes cellpose output data and converts it into Detectron2 format

    Inputs:
        - experiment: Name of experiment that was manually segmented
        - imgType: Name of folder containing images (phaseContrast, composite, etc.)
        - stage: Flag for making training or testing sets
    
    Outputs:
        - datasetDicts: Detectron2 segmentation format
    """
    cwd = os.path.dirname(__file__)
    dataPath = os.path.join(cwd, '../../data')

    # Get manual segmentations from cellPose
    segDir = os.path.join(dataPath,experiment,'segmentations', 'manual')
    segFiles = os.listdir(segDir)
    segFiles = [segFile for segFile in segFiles if segFile.endswith('.npy')]
    idx = 0

    # Split data depending on desired training/testing
    if stage in ['train', 'test']:
        print(f'In {stage} stage')
        random.seed(1234)
        random.shuffle(segFiles)
        trainPercent = 0.9
        trainNum = int(trainPercent*len(segFiles))
        if stage == 'train':
            segFiles = segFiles[0:trainNum]
        elif stage == 'test':
            segFiles = segFiles[trainNum:]

    datasetDicts = []
    for segFile in tqdm(segFiles):

        # Load in cellpose output
        segFull = os.path.join(segDir, segFile)
        seg = np.load(segFull, allow_pickle=True)
        seg = seg.item()

        # Split masks into 16 portions
        splitMasks = imSplit(seg['masks'], nIms = 16)
        nSplits = len(splitMasks)

        splitDir = f'split{nSplits}'
        imgBase = getImageBase(segFile)

        # For every split image, load corresponding image
        # Covert to Detectron2 Dataset format
        for splitNum in range(1, len(splitMasks)+1):
            imgFile = f'{imgType}_{imgBase}_{splitNum}.png'
            imgPath = os.path.join(dataPath, experiment, splitDir, imgType, imgFile)
                
            record = {}
            record['file_name'] = imgPath
            record['image_id'] = idx
            record['height'] = splitMasks[splitNum-1].shape[0]
            record['width'] = splitMasks[splitNum-1].shape[1]

            mask = splitMasks[splitNum-1]
            cellNums = np.unique(mask)
            cellNums = cellNums[cellNums != 0]

            cells = []
            # For each cell, convert to a polygon representation
            for cellNum in cellNums:
                contours = measure.find_contours(img_as_float(mask==cellNum), .5)
                fullContour = np.vstack(contours)

                px = fullContour[:,1]
                py = fullContour[:,0]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                # if len(poly) < 4:
                #     return
                cell = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                cells.append(cell)
            record["annotations"] = cells  
            datasetDicts.append(record)
            idx+=1        
    return datasetDicts
        
def addCellsToCatalog(experiment, name, imgType = 'phaseContrast'):
    trainName = f'{name}_train'
    testName = f'{name}_test'
    if trainName in DatasetCatalog:
        DatasetCatalog.remove(trainName)
        print('Removing training')
    if testName in DatasetCatalog:
        DatasetCatalog.remove(testName)
        print('Removing testing')
    inputs = [experiment, imgType, 'train']

    DatasetCatalog.register(trainName, lambda x=inputs: cellpose2Detectron(inputs[0], inputs[1], 'train'))
    MetadataCatalog.get(trainName).set(thing_classes=["cell"])

    DatasetCatalog.register(testName, lambda x=inputs: cellpose2Detectron(inputs[0], inputs[1], 'test'))
    MetadataCatalog.get(testName).set(thing_classes=["cell"])



