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
maxAmt      = 64
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
notes = 'Test with sgd optimizer'

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
plt.imshow(inputs[35].numpy().transpose((1,2,0)))
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
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
def data_dict(datasetDicts):
    return datasetDicts
inputs = [datasetDicts]
# DatasetCatalog.register('test', lambda x=inputs: data_dict(inputs[0]))
MetadataCatalog.get('test').set(thing_classes=[0, 1])
convert_to_coco_json('test', output_file='./test.json', allow_cached=False)
# %%
from detectron2.data.datasets import load_coco_json
x = load_coco_json('./test.json', '.')
# %%
