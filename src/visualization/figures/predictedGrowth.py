# %% [markdown]
"""
Residual plot of predicted vs actual growth
"""
# %%
from src.models.trainBB import singleCellLoader
from src.models import testBB

from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
# %%
homePath = Path('../../../')
datasetDictPath = homePath / './data/TJ2201/split16/TJ2201DatasetDictNoBorder.npy'
datasetDicts = np.load(datasetDictPath, allow_pickle=True)
# %%
wells = []
wells = [seg['file_name'].split('_')[1] for seg in datasetDicts]
uniqueWells, wellCts = np.unique(wells, return_counts=True)
# %%
wellsSave = ['E2', 'D2', 'E7']
datasetDictsSub = [seg for seg in datasetDicts if seg['file_name'].split('_')[1] in wellsSave]
# %%
modelName = 'classifySingleCellCrop-688020'
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
modelPath = Path.joinpath(homePath, 'models', 'classification', f'{modelName}.pth')
model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(modelPath, map_location=device))
model.eval()

num_ftrs = model.fc.in_features

# Create new layer and assign in to the last layer
# Number of output layers is now 2 for the 2 classes
model.fc = nn.Linear(num_ftrs, 2)


model.load_state_dict(torch.load(modelPath, map_location=device))
model.eval()

outPath = Path.joinpath(homePath, 'results', 'classificationTraining', f'{modelName}.out')

modelDetails = testBB.getModelDetails(outPath)

# %%
experiment  = 'TJ2201'
nIncrease   = 65
maxAmt      = 15000
batch_size  = 40
num_epochs  = 40
modelType   = 'resnet152'

dataPath = Path(homePath / f'data/{experiment}/raw/phaseContrast')

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        # transforms.Resize(356),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ]),
}

image_dataset = singleCellLoader(datasetDicts, 
                                data_transforms['train'], 
                                dataPath, 
                                nIncrease = modelDetails['nIncrease'], 
                                phase='none', 
                                maxAmt = modelDetails['maxAmt'])
dataset_sizes = len(image_dataset)
dataloader = DataLoader(image_dataset, batch_size=modelDetails['batch_size'], shuffle=True)


# %%
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
probs = []
allLabels = []
scores = []
running_corrects = 0
for inputs, labels in tqdm(dataloader, position=0, leave=True):
    # I have no idea why you have to do this but...
    # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/9
    inputs = inputs.float()
    inputs = inputs.to(device)
    labels = labels.to(device)


    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    probs.append(outputs.cpu().data.numpy())
    allLabels.append(labels.cpu().data.numpy())
    scores.append(F.softmax(outputs, dim=1).cpu().data.numpy())
    running_corrects += torch.sum(preds == labels.data)
    

probs = np.concatenate(probs)
allLabels = np.concatenate(allLabels)
scores = np.concatenate(scores)

modelTestResults = {'probs': probs, 'allLabels': allLabels, 'scores': scores, 'images': dataloader.dataset.imgNames}
pickle.dump(modelTestResults, open(homePath / 'results/classificationResults/TJ2201PredGrowth.pickle', "wb"))