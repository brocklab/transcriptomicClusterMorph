# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.models import testBB, trainBB
from src.data.fileManagement import convertDate, getModelDetails
from src.models import modelTools
from pathlib import Path
import numpy as np
import time
import sys
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import models
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim


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
'notes'         : notes,
'augmentation'  : None

}
experiment = 'TJ2310'
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
dataloader, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath,
                                               modelInputs,
                                               phase=['none']
                                            )
# %%

# %%
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
modelName = 'classifySingleCellCrop-1682975256'
homePath = Path('../')
modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')
if not outPath.exists():
    outPath = Path(str(outPath).replace('.out', '.txt'))
assert outPath.exists(), outPath
modelDetails = getModelDetails(outPath)
print(modelDetails)
model = trainBB.getTFModel(modelDetails['modelType'], modelPath)

dataPath = Path.joinpath(homePath, 'data', modelDetails['experiment'], 'raw', 'phaseContrast')

dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                            dataPath,
                                            modelDetails,
                                            phase = ['none'],
                                            isShuffle = False
                                            )
# %%
inputs, labels = next(iter(dataloader))
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
model.to(device)
# %%
allPreds = []
for inputs, labels in tqdm(dataloader):
    inputs = inputs.float()
    inputs = inputs.to(device)
    labels = labels.to(device)


    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    allPreds.append(preds.cpu().numpy())
    # print(sum(preds)/len(preds))
# %%
preds = np.concatenate(allPreds)
print(sum(preds)/len(preds))
# %%
