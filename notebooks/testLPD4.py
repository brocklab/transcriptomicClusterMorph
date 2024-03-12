# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.models import testBB, trainBB
from src.data.fileManagement import convertDate, getModelDetails,  loadSegmentationJSON
from src.models import modelTools
from src.visualization.segmentationVis import viewPredictorResult
from pathlib import Path

import numpy as np
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim



# %%
experiment  = 'TJ2303-LPD4'
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
'notes'         : notes,
'augmentation'  : None

}

# %%
# datasetDictPath = Path(f'../data/{experiment}/{experiment}Segmentations.json')
# datasetDicts = loadSegmentationJSON(datasetDictPath)
datasetDicts = np.load('../data/TJ2303-LPD4/TJ2303-LPD4DatasetDicts.npy', allow_pickle=True)
# %%
datasetDictPath = Path(f'../data/{experiment}/{experiment}DatasetDicts.npy')
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
# %%
inputs, classes = next(iter(dataloader))
# %%
plt.imshow(inputs[20].numpy().transpose((1,2,0)))
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
modelName = 'classifySingleCellCrop-1700026902'
modelName = 'classifySingleCellCrop-1700187095'
modelName = 'classifySingleCellCrop-1709878770'

homePath = Path('../')
modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
if not outPath.exists():
    outPath = Path(str(outPath).replace('.out', '.txt'))
assert outPath.exists(), outPath
modelInputs = getModelDetails(outPath)
modelInputs['augmentation'] = None
print(modelInputs)
model = trainBB.getTFModel(modelInputs['modelType'], modelPath)

dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

modelInputs['experiment'] = 'TJ2310'

allWells, allProps = [], []
for testWell in wellSize.keys():
    modelInputs['testWell'] = testWell
    dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                                dataPath,
                                                modelInputs,
                                                isShuffle = False
                                                )

    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model.to(device)
    allPreds = []
    for inputs, labels in tqdm(dataloaders['test']):
        inputs = inputs.float()
        inputs = inputs.to(device)
        labels = labels.to(device)


        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        allPreds.append(preds.cpu().numpy())
        # print(sum(preds)/len(preds))
    preds = np.concatenate(allPreds)
    prop = sum(preds)/len(preds)*100

    allWells.append(testWell)
    allProps.append(prop)
    pd.DataFrame([allWells, allProps]).to_csv('./lpd4Pred.csv')

    print(f'Test well: {testWell} = {prop:0.2f}%')
# %%