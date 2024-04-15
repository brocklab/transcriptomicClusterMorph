# %%
from src.models.trainBB import makeImageDatasets, train_model, getTFModel
from src.models import modelTools
from pathlib import Path
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm 

from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.optim as optim

from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode

# %% Add argparse
parser = argparse.ArgumentParser(description='Network prediction parameters')
parser.add_argument('--experiment', type = str, metavar='experiment',  help = 'Experiment to run')
parser.add_argument('--nIncrease',  type = int, metavar='nIncrease',   help = 'Increase of bounding box around cell')
parser.add_argument('--maxAmt',     type = int, metavar='maxAmt',      help = 'Max amount of cells')
parser.add_argument('--batch_size', type = int, metavar='batch_size',  help = 'Batch size')
parser.add_argument('--num_epochs', type = int, metavar='num_epochs',  help = 'Number of epochs')
parser.add_argument('--modelType',  type = str, metavar='modelType',   help = 'Type of model (resnet, vgg, etc.)')
parser.add_argument('--notes',      type = str, metavar='notes',       help = 'Notes on why experiment is being run')
parser.add_argument('--optimizer',  type = str, metavar='optimizer',   help = 'Optimizer type')
parser.add_argument('--augmentation',  type = str, metavar='augmentation',   help = 'Image adjustment (None, blackoutCell, stamp)')
parser.add_argument('--nIms',       type = int, metavar='augmentation',   help = 'Number of images the initial full image was split into (experiment dependent). 20x magnification: 16, 10x magnification: 4')
parser.add_argument('--maxImgSize', type = int, metavar='maxImgSize', help = 'The final size of the image. If larger than the bounding box, pad with black, otherwise resize the image')

# This is for running the notebook directly
args, unknown = parser.parse_known_args()

# %%
experiment  = 'TJ2453-436Co'
nIncrease   = 25
maxAmt      = 200000000000
batch_size  = 64
num_epochs  = 32
modelType   = 'resnet152'
optimizer = 'sgd'
augmentation = None
nIms = 16
maxImgSize = 300
notes = 'Run on coculture wells only'

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
'optimizer'     : optimizer, 
'augmentation'  : augmentation,
'nIms'          : nIms,
'maxImgSize'    : maxImgSize
}

argItems = vars(args)

for item, value in argItems.items():
    if value is not None:
        print(f'Replacing {item} value with {value}')
        modelInputs[item] = value
modelDetailsPrint = modelTools.printModelVariables(modelInputs)
# %%
def convertRecords(datasetDicts):
    newDatasetDicts = []
    for record in tqdm(datasetDicts):
        well = record['file_name'].split('_')[1]
        if well.endswith('10') or well.endswith('11'):
            continue
        record = record.copy()

        for annotation in record['annotations']:
            annotation['bbox'] = BoxMode.convert(annotation['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS)
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
        newDatasetDicts.append(record)
    return newDatasetDicts
# %%
datasetDicts = load_coco_json(f'../data/{experiment}/{experiment}SegmentationsFiltered.json', '.')
datasetDicts = convertRecords(datasetDicts)
# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')

dataloaders, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath,
                                               modelInputs
                                            )
np.unique(dataloaders['train'].dataset.phenotypes, return_counts=True)
# %%
inputs, classes = next(iter(dataloaders['train']))
plt.subplot(121)
plt.imshow(inputs[0].numpy().transpose((1,2,0)))
plt.subplot(122)
plt.imshow(inputs[1].numpy().transpose((1,2,0)))
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
model2 = nn.DataParallel(model)

# %%
modelDetailsPrint = modelTools.printModelVariables(modelInputs)


with open(resultsSaveName, 'a') as file:
    file.write(modelDetailsPrint)

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
                    num_epochs=modelInputs['num_epochs']
                    )
