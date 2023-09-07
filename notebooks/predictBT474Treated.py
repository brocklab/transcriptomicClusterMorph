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

# This is for running the notebook directly
args, unknown = parser.parse_known_args()
# %%
experiment      = 'TJ2310 and TJ2302'
nIncrease       = 10
maxAmt          = 9e9
batch_size      = 64
num_epochs      = 32
modelType       = 'resnet152'
optimizer       = 'sgd'
augmentation    = None
nIms            = 16
maxImgSize      = 150
notes = 'TJ2310 is LPD treated and TJ2302 is pretreatment + lineage recall (red)'
# %%
experiment = 'TJ2310'
dataPath = Path(f'../data/{experiment}/split4/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/{experiment}DatasetDicts-1.npy')
datasetDicts = list(np.load(datasetDictPath, allow_pickle=True))
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')
# %%
experiment = 'TJ2302'
dataPath = Path(f'../data/{experiment}/split4/phaseContrast')
datasetDictPath = Path(f'../data/{experiment}/{experiment}DatasetDicts-0.npy')
datasetDicts = list(np.load(datasetDictPath, allow_pickle=True))
dataPath = Path(f'../data/{experiment}/raw/phaseContrast')