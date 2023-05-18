# %%
from src.data import fileManagement
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.data.fileManagement import convertDate
from src.models import modelTools
from pathlib import Path
import numpy as np
import time
import sys
import datetime

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim

# %%
experiment  = 'TJ2201'
nIncrease   = 25
maxAmt      = 10000000
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
optimizer = 'sgd'
notes = 'Early vs late timepoints using 3 days as starting point'

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
'optimizer'     : optimizer
}
# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorderFull.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
co = ['B7','B8','B9','B10','B11','C7','C8','C9','C10','C11','D7','D8','D9','D10','D11','E7','E8','E9','E10','E11']
datasetDicts = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in co]
datasetDicts = [record for record in datasetDicts if len(record['annotations']) > 0]

# %% Check dates
dates = []
record = datasetDicts[0]
for record in datasetDicts:
    strDate = '_'.join(record['file_name'].split('_')[3:5])
    dates.append(fileManagement.convertDate(strDate))

# Get midpoint
firstDate = min(dates)
timeDiff = max(dates) - firstDate
midPoint = timeDiff.total_seconds()/2
day3Seconds = 3*60*60*24

datasetDictsLate, datasetDictsEarly = [], []
for record in datasetDicts:
    strDate = '_'.join(record['file_name'].split('_')[3:5])
    date = fileManagement.convertDate(strDate)
    timeDiff = date - firstDate
    timeDiff = day3Seconds - timeDiff.total_seconds()
    # Find if before or after given point
    if timeDiff > 0:
        datasetDictsEarly.append(record)
    elif timeDiff <= 0:
        datasetDictsLate.append(record)

# %%
dataloaders, dataset_sizes = makeImageDatasets(datasetDictsLate, 
                                               dataPath,
                                               modelInputs
                                            )
np.unique(dataloaders['train'].dataset.phenotypes, return_counts=True)
# %%
inputs, classes = next(iter(dataloaders['train']))
# %%
# plt.imshow(inputs[16].numpy().transpose((1,2,0)))
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not modelSaveName.parent.exists():
    raise NotADirectoryError('Model directory not found')

model = getTFModel(modelInputs['modelType'])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
modelDetailsPrint = modelTools.printModelVariables(modelInputs)

with open(resultsSaveName, 'a') as file:
    file.write(modelDetailsPrint)

with open(resultsSaveName, 'a') as file:
    file.write('\n Late images \n')
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
dataloaders, dataset_sizes = makeImageDatasets(datasetDictsEarly, 
                                               dataPath,
                                               modelInputs
                                            )
np.unique(dataloaders['train'].dataset.phenotypes, return_counts=True)
# %%
inputs, classes = next(iter(dataloaders['train']))
# %%
# plt.imshow(inputs[16].numpy().transpose((1,2,0)))
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not modelSaveName.parent.exists():
    raise NotADirectoryError('Model directory not found')

model = getTFModel(modelInputs['modelType'])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
with open(resultsSaveName, 'a') as file:
    file.write('\n Early images \n')

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