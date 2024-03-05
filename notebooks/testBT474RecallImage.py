# %%
from src.models.trainBB import makeImageDatasets, getTFModel
from src.models import testBB, trainBB
from src.data.fileManagement import getModelDetails
from src.models import modelTools
from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import torch.nn as nn
import torch
import torch.optim as optim

from detectron2.data.datasets import load_coco_json
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

experiment = 'TJ24XX'

modelInputs = {

'experiment'    : experiment, 
'nIncrease'     : nIncrease,
'maxAmt'        : maxAmt,
'batch_size'    : batch_size,
'num_epochs'    : num_epochs,
'modelType'     : modelType,
'modelName'     : modelSaveName.parts[-1],
'modelIDSource' : idSource,
'notes'         : notes,
'augmentation'  : None

}
# %%
import json
experimentParamsLoc = '/home/user/work/cellMorph/data/experimentParams.json'
with open(experimentParamsLoc, 'r') as json_file:
    experimentParams = json.load(json_file)
experimentParams[experiment] = experimentParams['TJ2303-LPD4'].copy()
experimentJson = json.dumps(experimentParams)
with open(experimentParamsLoc, 'w') as json_file:
    json_file.write(experimentJson)
# %%
datasetDicts = load_coco_json('../data/TJ2442F/TJ2442FSegmentations.json', '.')

dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
# %%
wellSize = {}
for seg in datasetDicts:
    well = seg['file_name'].split('_')[1]
    if well not in wellSize.keys():
        wellSize[well] = 0
    wellSize[well] += len(seg['annotations'])
# %%
dataloader, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath,
                                               modelInputs,
                                               phase=['none']
                                            )