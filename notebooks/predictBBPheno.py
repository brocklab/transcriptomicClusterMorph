# %%
# %load_ext autoreload
# %autoreload 2

# %%
from src.models.trainPhenoPredBB import makeImageDatasets, train_model
from src.models import modelTools
from pathlib import Path
import numpy as np
import time
import sys

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim

# %%
experiment  = 'TJ2201'
nIncrease   = 55
maxAmt      = 15000
batch_size  = 40
num_epochs  = 40
modelType   = 'resnet152'

modelID, idSource = modelTools.getModelID(sys.argv)
modelSaveName = Path(f'../models/classifySingleCellCrop-{modelID}.pth')

modelInputs = {

'experiment'    : experiment, 
'nIncrease'     : nIncrease,
'maxAmt'        : maxAmt,
'batch_size'    : batch_size,
'num_epochs'    : num_epochs,
'modelType'     : modelType,
'modelName'     : modelSaveName.parts[-1],
'modelIDSource' : idSource

}

modelTools.printModelVariables(modelInputs)
# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/split16/{experiment}DatasetDictNoBorder.npy')
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %%
dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath, 
                                               nIncrease=nIncrease, 
                                               maxAmt = maxAmt, 
                                               batch_size=batch_size
                                               )
# %%
dataset_sizes

# %%
inputs, classes = next(iter(dataloaders['train']))
# %%
model = models.resnet152(pretrained=True)

if not modelSaveName.parent.exists():
    raise NotADirectoryError('Model directory not found')
    
num_ftrs = model.fc.in_features
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create new layer and assign in to the last layer
# Number of output layers is now 2 for the 2 classes
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

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
                    num_epochs=num_epochs
                    )
# %%
