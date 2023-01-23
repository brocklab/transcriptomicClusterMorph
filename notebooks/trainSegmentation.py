# %% [markdown]
"""
This will take cellpose results and train a segmentation model on them
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

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
# %%
experiment = 'TJ2201'
stage = 'train'
imgType = 'phaseContrast'

# Get manual segmentations from cellPose
segDir = os.path.join('../data',experiment,'segmentations', 'manual')
segFiles = os.listdir(segDir)
segFiles = [segFile for segFile in segFiles if segFile.endswith('.npy')]
idx = 0
# Split data depending on desired 
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

    splitMasks = imSplit(seg['masks'])
    nSplits = len(splitMasks)

    splitDir = f'{experiment}Split{nSplits}'
    imgBase = getImageBase(segFile)    
    for splitNum in range(1, len(splitMasks)+1):
        imgFile = f'{imgType}_{imgBase}_{splitNum}.png'
        imgPath = os.path.join('../data', splitDir, imgType, imgFile)
        assert os.path.isfile(imgPath)
        record = {}
        record['file_name'] = imgPath
        record['image_id'] = idx
        record['height'] = splitMasks[splitNum-1].shape[0]
        record['width'] = splitMasks[splitNum-1].shape[1]

        mask = splitMasks[splitNum-1]
        cellNums = np.unique(mask)
        cellNums = cellNums[cellNums != 0]

        cells = []
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
        
        
        
# %%
