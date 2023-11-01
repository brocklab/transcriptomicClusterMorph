# %%
from src.data.fileManagement import collateModelParameters

from src.models import testBB, trainBB
from src.data.fileManagement import splitName2Whole
from src.data.imageProcessing import bbIncrease, bbIncreaseBlackout

from pathlib import Path
import numpy as np
import sys
from tqdm import tqdm
import pickle
import os
import random 
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

import detectron2.data.datasets as datasets
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2
from detectron2.structures import BoxMode
# %%
dfExperiment = collateModelParameters(generate=True)
# %%
experiment = 'TJ2201'
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
datasetDicts = [record for record in datasetDicts if len(record['annotations']) > 0]
# %%
modelDict = {   
            'No Augmentation': 'classifySingleCellCrop-727592',
            'No Surrounding' : 'classifySingleCellCrop-1693286401',
            'No Texture'     : 'classifySingleCellCrop-1693256617'
}

resDict = {}
for augName, modelName in tqdm(modelDict.items()):

    modelPath = str(Path('../models/classification') / f'{modelName}.pth')
    resPath =   str(Path('../results/classificationTraining') / f'{modelName}.txt')
    modelInputs = testBB.getModelDetails(resPath)

    if 'augmentation' not in modelInputs.keys():
        modelInputs['augmentation'] = None

    batch_size   = modelInputs['batch_size']
    dataloaders, dataset_sizes = trainBB.makeImageDatasets(
                                                datasetDicts, 
                                                dataPath,
                                                modelInputs
                                                )
    model = trainBB.getTFModel(modelInputs['modelType'], modelPath)
    probs, allLabels, scores = testBB.testModel(model, dataloaders, mode = 'test')
    res = testBB.testResults(probs, allLabels, scores, modelName)

    resDict[augName] = res
# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for augName, res in resDict.items():
    auc = res.auc
    plotLabel = f'{augName} AUC = {auc:0.2f}'
    plt.plot(res.fpr, res.tpr, label=plotLabel, linewidth=3)


plt.title('ESAM (+/-) Cell Classification')
plt.legend(fontsize=12, loc='lower right')

# %%
experiment = 'TJ2201'
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDicts = np.load(datasetDictPath, allow_pick    # dataloaders, dataset_sizes = trainBB.makeImageDatasets(
    #                                             datasetDicts, 
    #                                             dataPath,
    #                                             modelInputs
    #                                             )
modelName = 'classifySplitCoculture-1684861223'
modelPath = str(Path('../models/classification') / f'{modelName}.pth')
resPath =   str(Path('../results/classificationTraining') / f'{modelName}.txt')
modelInputs = testBB.getModelDetails(resPath)

if 'augmentation' not in modelInputs.keys():
    modelInputs['augmentation'] = None

modelInputs['maxAmt'] = 10000

batch_size   = modelInputs['batch_size']
dataloaders, dataset_sizes = trainBB.makeImageDatasets(
                                            datasetDicts, 
                                            dataPath,
                                            modelInputs
                                            )
model = trainBB.getTFModel(modelInputs['modelType'], modelPath)
probs, allLabels, scores = testBB.testModel(model, dataloaders, mode = 'test')
res = testBB.testResults(probs, allLabels, scores, modelName)
