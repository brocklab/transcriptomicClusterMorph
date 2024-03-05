# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm

from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
import detectron2

from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.models import testBB
from src.data.fileManagement import convertDate
from src.models import modelTools
from pathlib import Path
import numpy as np
import time
import sys
import datetime
import matplotlib.pyplot as plt

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from src.models import trainBB
from src.data.fileManagement import getModelDetails

# %%
def getGreenRecord(datasetDicts, datasetDictsGreen = []):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []

        for annotation in record['annotations']:
            annotation['bbox'] = detectron2.structures.BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            if annotation['category_id'] == 1:
                newAnnotations.append(annotation)
        if len(newAnnotations) > 0:
            record['annotations'] = newAnnotations
            datasetDictsGreen.append(record)
    return datasetDictsGreen
# for record in tqdm(datasetDictsTreat):
#     for cell in record['annotations']:
#         cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
#         cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%
datasetDictsGreen = {}

datasetDicts = load_coco_json('../data/TJ2342A/TJ2342ASegmentations.json', '.')
datasetDictsGreen['TJ2342A'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442D/TJ2442DSegmentations.json', '.')
datasetDictsGreen['TJ2442D'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442E/TJ2442ESegmentations.json', '.')
datasetDictsGreen['TJ2442E'] = getGreenRecord(datasetDicts, [])

datasetDicts = load_coco_json('../data/TJ2442F/TJ2442FSegmentations.json', '.')
datasetDictsGreen['TJ2442F'] = getGreenRecord(datasetDicts, [])
# %%
import json
experimentParamsLoc = '/home/user/work/cellMorph/data/experimentParams.json'
with open(experimentParamsLoc, 'r') as json_file:
    experimentParams = json.load(json_file)
for experiment in datasetDictsGreen.keys():
    experimentParams[experiment] = experimentParams['TJ2303-LPD4'].copy()
experimentJson = json.dumps(experimentParams)
with open(experimentParamsLoc, 'w') as json_file:
    json_file.write(experimentJson)
# %% Try to find dead cells
from skimage.measure import label, regionprops, regionprops_table
from skimage.draw import polygon2mask

def getAreaEcc(polygon, imageShape):
    polyx = polygon[::2]
    polyy = polygon[1::2]
    polygonSki = list(zip(polyy, polyx))
    mask = polygon2mask(imageShape, polygonSki)
    reg = regionprops(mask.astype(np.uint8))

    area = reg[0].area
    eccentricity = reg[0].eccentricity
    
    return area, eccentricity

allArea, allEcc, imgNames, allPoly = [], [], [], []
for experiment in datasetDictsGreen:
    datasetDicts = datasetDictsGreen[experiment]
    nAnnos = 0
    for record in datasetDicts:
        nAnnos += len(record['annotations'])
    print(f'{experiment} had {nAnnos} cells identified')
for experiment in datasetDictsGreen.keys():
    datasetDicts = datasetDictsGreen[experiment]
    print(experiment)
    nAnnos = 0
    for record in tqdm(datasetDicts):
        image_shape = [record['height'], record['width']]
        newAnnotations = []
        for annotation in record['annotations']:
            segmentation = annotation['segmentation'][0]

            area, ecc = getAreaEcc(segmentation, image_shape)

            if area < 1000 and ecc < 0.3:
                newAnnotations.append(annotation)
            allArea.append(area)
            allEcc.append(ecc)
            allPoly.append(segmentation)
            imgNames.append(record['file_name'])
        nAnnos += len(newAnnotations)
        record['annotations'] = newAnnotations

    print(f'{experiment} had {nAnnos} cells identified')
# %%
