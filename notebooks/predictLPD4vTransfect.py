# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from pathlib import Path
from tqdm import tqdm
import argparse 

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
def getPhaseRecord(datasetDicts, newDatasetDicts = []):
    for record in tqdm(datasetDicts):
        record = record.copy()
        newAnnotations = []

        for annotation in record['annotations']:
            annotation['bbox'] = detectron2.structures.BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
            if annotation['category_id'] == 0:
                annotation['category_id'] = 1
                newAnnotations.append(annotation)
        if len(newAnnotations) > 0:
            record['annotations'] = newAnnotations
            newDatasetDicts.append(record)
    return newDatasetDicts

# for record in tqdm(datasetDictsTreat):
#     for cell in record['annotations']:
#         cell['bbox'] = detectron2.structures.BoxMode.convert(cell['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
#         cell['bbox_mode'] = BoxMode.XYXY_ABS
# %%

datasetDicts = load_coco_json('../data/TJ2342A/TJ2342ASegmentations.json', '.')
datasetDictsTransfect = getPhaseRecord(datasetDicts, [])
# %%
datasetDictsLPD4 = np.load('../data/TJ2303-LPD4/TJ2303-LPD4DatasetDicts.npy', allow_pickle=True)

# %%
experiment  = 'TJ2302'
nIncrease   = 20
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
experiment = 'TJ2342A'
trainLoaders, testLoaders = [], []
modelInputs['experiment'] = experiment
modelInputs['maxAmt'] = 5000
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

dataloaders, dataset_sizes = makeImageDatasets(datasetDictsTransfect, 
                                            dataPath,
                                            modelInputs,
                                            phase = ['train', 'test']
                                            )

trainLoaders.append(dataloaders['train'].dataset)
testLoaders.append(dataloaders['test'].dataset)
# %%
experiment = 'TJ2342A'
trainLoaders, testLoaders = [], []
modelInputs['experiment'] = experiment
modelInputs['maxAmt'] = 10000
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

dataloaders, dataset_sizes = makeImageDatasets(datasetDictsTransfect, 
                                            dataPath,
                                            modelInputs,
                                            phase = ['none']
                                            )
transfectTrain = Subset(dataloaders.dataset, range(0, 7000))
transfectTest = Subset(dataloaders.dataset, range(7000, 10000))

# %%
dataPath = Path(f'../data/TJ2303-LPD4/raw/phaseContrast')
modelInputs['experiment'] = 'TJ2303-LPD4'
from torch.utils.data import Subset
dataloaders, dataset_sizes = makeImageDatasets(datasetDictsLPD4, 
                                               dataPath,
                                               modelInputs,
                                               phase=['none']
                                            )
lpd4Train = Subset(dataloaders.dataset, range(0, 7000))
lpd4Test = Subset(dataloaders.dataset, range(7000, 10000))

# %%
trainLoader = DataLoader(ConcatDataset([transfectTrain, lpd4Train]),
                            batch_size = modelInputs['batch_size'],
                            shuffle = True)

testLoader = DataLoader(ConcatDataset([transfectTest, lpd4Test]),
                            batch_size = modelInputs['batch_size'],
                            shuffle = True)
# %%
inputs, classes = next(iter(trainLoader))
# %%
dataloaders = {'train': trainLoader, 'test': testLoader}
dataset_sizes = {'train': len(trainLoader.dataset), 'test': len(testLoader.dataset)}
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not modelSaveName.parent.exists():
    raise NotADirectoryError('Model directory not found')

model = getTFModel(modelInputs['modelType'])
model.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
model2 = nn.DataParallel(model)

# %%
# Scheduler to update lr
# Every 7 epochs the learning rate is multiplied by gamma
setp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, 
                    criterion, 
                    optimizer, 
                    setp_lr_scheduler,
                    dataloaders,
                    dataset_sizes, 
                    modelSaveName,
                    resultsSaveName,
                    num_epochs=num_epochs
                    )

# %%
