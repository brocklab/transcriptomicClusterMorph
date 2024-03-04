# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm

from detectron2.data.datasets import load_coco_json


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

# %%
def getGreenRecord(datasetDicts, datasetDictsGreen = []):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []
        for annotation in record['annotations']:
            if annotation['category_id'] == 1:
                newAnnotations.append(annotation)
        if len(newAnnotations) > 0:
            record['annotations'] = newAnnotations
            datasetDictsGreen.append(record)
    return datasetDictsGreen

# %%
datasetDictsGreen = {}

datasetDicts = load_coco_json('../data/TJ2442D/TJ2442DSegmentations.json', '.')
datasetDictsGreen['TJ2442D'] = getGreenRecord(datasetDicts)
datasetDicts = load_coco_json('../data/TJ2442E/TJ2442ESegmentations.json', '.')
datasetDictsGreen['TJ2442E'] = getGreenRecord(datasetDicts)
datasetDicts = load_coco_json('../data/TJ2442F/TJ2442FSegmentations.json', '.')
datasetDictsGreen['TJ2442F'] = getGreenRecord(datasetDicts)
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
# %%
experiment  = 'TJ2302'
nIncrease   = 10
maxAmt      = 9e9
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
notes = 'Full test with Adam'

modelID, idSource = modelTools.getModelID(sys.argv)
modelSaveName = Path(f'../models/classification/classifySingleCellCrop-{modelID}.pth')
resultsSaveName = Path(f'../results/classificationTraining/classifySingleCellCrop-{modelID}.txt')
modelInputs = {

'experiment'    : experiment, 
'nIncrease'     : nIncrease,
'maxAmt'        : maxAmt,
'batch_size'    : batch_size,
'num_epochs'    : num_epochs,
'modelType'     : modelType,
'modelName'     : modelSaveName.parts[-1],
'modelIDSource' : idSource,
'notes'         : notes

}

# %%
experiments = datasetDictsGreen.keys()
trainLoaders, testLoaders = [], []
for experiment in experiments:
    modelInputs['experiment'] = experiment
    dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

    dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                                dataPath,
                                                modelInputs
                                                )
    
    trainLoaders.append(dataloaders['train'].dataset)
    testLoaders.append(dataloaders['test'].dataset)
# %%
trainLoaders = DataLoader(ConcatDataset(trainLoaders),
                             batch_size = modelInputs['batch_size'],
                             shuffle = True)

testLoaders = DataLoader(ConcatDataset(testLoaders),
                             batch_size = modelInputs['batch_size'],
                             shuffle = True)
# %%
# %%
x = DataLoader(ConcatDataset([dataloaders['train'].dataset, dataloaders['test'].dataset]),
               batch_size = batch_size,
               shuffle = False
               )
# %%
inputs, classes = next(iter(trainLoaders))