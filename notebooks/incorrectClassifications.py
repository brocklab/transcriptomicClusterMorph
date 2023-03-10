# %% [markdown]
"""
This is a notebook to try and determine if there is a systematic reason that
231 cells were incorrectly identified. 
"""
# %%
from src.models.trainBB import singleCellLoader, getTFModel
from src.data.fileManagement import convertDate, splitName2Whole, collateModelParameters

from src.models.testBB import getModelDetails
from src.models import trainBB

from src.models import testBB
from src.visualization.trainTestRes import plotTrainingRes
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle 
from tqdm import tqdm
import pandas as pd
from scipy import stats
from skimage.io import imread

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torch.nn.functional as F
# %%
# First get training/testing results
homePath = Path('../')
modelPath = homePath / 'models' / 'classification'
modelNames = list(modelPath.iterdir())
modelNames = [str(modelName.parts[-1]).split('.')[0] for modelName in modelNames]
modelNames.sort()
datasetDictPathFull = homePath / 'data/TJ2201/split16/TJ2201DatasetDictNoBorderFull.npy'
datasetDictPathPartial = homePath / 'data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'

# %%
datasetDicts = np.load(datasetDictPathPartial, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]

# %%
modelName = 'classifySingleCellCrop-727592'
modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
modelDetails = getModelDetails(outPath)

model = trainBB.getTFModel(modelDetails['modelType'], modelPath)



dataPath = Path.joinpath(homePath, 'data', modelDetails['experiment'], 'raw', 'phaseContrast')

dataloaders, dataset_sizes = trainBB.makeImageDatasets(datasetDicts,
                                            dataPath, 
                                            nIncrease    = modelDetails['nIncrease'], 
                                            maxAmt       = modelDetails['maxAmt'], 
                                            batch_size   = modelDetails['batch_size'],
                                            isShuffle = False
                                            )
# %%
segs = dataloaders['test'].dataset.segmentations
imgNames = dataloaders['test'].dataset.imgNames
imgDir = homePath / 'data' / 'TJ2201' / 'split16' / 'composite'
# %%
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
probs = []
allLabels = []
scores = []
running_corrects = 0
for inputs, labels in tqdm(dataloaders['test'], position=0, leave=True):
    # I have no idea why you have to do this but...
    # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/9
    inputs = inputs.float()
    inputs = inputs.to(device)
    labels = labels.to(device)


    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    probs.append(outputs.cpu().data.numpy())
    allLabels.append(labels.cpu().data.numpy())
    scores.append(F.softmax(outputs, dim=1).cpu().data.numpy())
    running_corrects += torch.sum(preds == labels.data)
    

probs = np.concatenate(probs)
allLabels = np.concatenate(allLabels)
scores = np.concatenate(scores)
preds = np.argmax(scores, axis= 1)
# %%
idx = np.random.randint(len(probs))
while True:
    pred = preds[idx]
    truth = allLabels[idx]
    if truth == pred:
        idx += 1
        continue
    else:
        imgName = imgNames[idx].replace('phaseContrast', 'composite')
        img = imread(imgDir / imgName)
        seg = segs[idx]
        score = scores[idx]
        plotTitle = f'{score}\nTruth: {truth} Pred: {pred}'

        plt.imshow(img)
        plt.plot(seg[:,0], seg[:,1])
        plt.title(plotTitle)
    break
        