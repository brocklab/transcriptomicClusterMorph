# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
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

# %%
experiment  = 'TJ2302'
nIncrease   = 10
maxAmt      = 50000
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
notes = 'Run only on coculture wells'

modelID, idSource = modelTools.getModelID(sys.argv)
modelSaveName = Path(f'../models/classification/classifySingleCellCrop-{modelID}.pth')

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

modelTools.printModelVariables(modelInputs)
# %%
dataPath = Path(f'../data/{experiment}/split4/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/{experiment}DatasetDicts-1.npy')
datasetDicts = list(np.load(datasetDictPath, allow_pickle=True))
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

# %%
wellSize = {}
for seg in datasetDicts:
    well = seg['file_name'].split('_')[1]
    if well not in wellSize.keys():
        wellSize[well] = 0
    wellSize[well] += len(seg['annotations'])
# %%
dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath, 
                                               nIncrease    = modelInputs['nIncrease'], 
                                               maxAmt       = modelInputs['maxAmt'], 
                                               batch_size   = modelInputs['batch_size']
                                               )
# %%
np.unique(dataloaders['train'].dataset.phenotypes, return_counts=True)
# %%
inputs, classes = next(iter(dataloaders['train']))
# %%
plt.imshow(inputs[35].numpy().transpose((1,2,0)))
# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not modelSaveName.parent.exists():
    raise NotADirectoryError('Model directory not found')

model = getTFModel(modelInputs['modelType'])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# %%
model2 = nn.DataParallel(model)

# %%
# Scheduler to update lr
# Every 7 epochs the learning rate is multiplied by gamma
setp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model2, 
                    criterion, 
                    optimizer, 
                    setp_lr_scheduler, 
                    dataloaders, 
                    dataset_sizes, 
                    modelSaveName, 
                    num_epochs=num_epochs
                    )
# %%
