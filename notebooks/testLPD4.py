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
import datetime
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

# %%
# datasetDictPath = Path(f'../data/{experiment}/{experiment}Segmentations.json')
# datasetDicts = loadSegmentationJSON(datasetDictPath)
datasetDicts = np.load('../data/TJ2303-LPD4/TJ2303-LPD4DatasetDicts.npy', allow_pickle=True)
# %%
wellSize = {}
for seg in datasetDicts:
    well = seg['file_name'].split('_')[1]
    if well not in wellSize.keys():
        wellSize[well] = 0
    wellSize[well] += len(seg['annotations'])
# %%
predictor = modelTools.getSegmentModel('../models/sartoriusBT474')
# %%
imPath = '/home/user/work/cellMorph/data/TJ2310/split4/phaseContrast/phaseContrast_D8_2_2023y05m25d_07h41m_1.png'
viewPredictorResult(predictor, imPath)
# %%
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
dataloader, dataset_sizes = makeImageDatasets(datasetDicts, 
                                               dataPath,
                                               modelInputs,
                                               phase=['none']
                                            )
# %%
inputs, labels = next(iter(dataloader))

# %%
