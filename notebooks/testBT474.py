# %%
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

modelDetailsPrint = modelTools.printModelVariables(modelInputs)
with open(resultsSaveName, 'a') as file:
    file.write(modelDetailsPrint)
# %%
dataPath = Path(f'../data/{experiment}/split4/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/{experiment}DatasetDicts-1-copy.npy')
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
                                               modelInputs
                                            )
                                            #    nIncrease    = modelInputs['nIncrease'], 
                                            #    maxAmt       = modelInputs['maxAmt'], 
                                            #    batch_size   = modelInputs['batch_size']
                                            #    )
# %%
np.unique(dataloaders['train'].dataset.phenotypes, return_counts=True)
# %%
inputs, classes = next(iter(dataloaders['train']))
# %%
# plt.imshow(inputs[35].numpy().transpose((1,2,0)))
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
probs, allLabels, scores, imgNames = testBB.getModelResults(modelName, homePath, datasetDicts, mode = 'test')
# %%
res = testBB.testResults(probs, allLabels, scores, imgNames, modelName)

# %%
plt.figure()
plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
auc = res.auc
plotLabel = f'BB increase AUC = {auc:0.2f}'
plt.plot(res.fpr, res.tpr, label=plotLabel, linewidth=3)
plt.legend(fontsize=12, loc='lower right')
plt.title('BT474 Lineage 1 Classification')
# %%